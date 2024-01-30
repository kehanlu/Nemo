from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel, SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.multimodal.speechllm.parts.utils.data_utils import align_feat_seq_list, get_nested_dict_value
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


# @kehan
from transformers import WhisperModel

__all__ = ["WhisperPerceptionModel"]


class PromptAdapter(NeuralModule):
    def __init__(self, cfg, whisper_config):
        super().__init__()
        self.prompt_size = cfg.perception.prompt_size
        self.layer_prompts = nn.ModuleList([
            nn.Embedding(self.prompt_size, whisper_config.d_model) for _ in range(whisper_config.encoder_layers)
        ])
        self.layer_weights = nn.Parameter(torch.zeros(self.prompt_size, whisper_config.encoder_layers, dtype=torch.float), requires_grad=True) # (prompt_size, encoder_layers)

        self.proj = nn.Sequential(
                nn.LayerNorm(whisper_config.d_model),
                nn.Linear(whisper_config.d_model, cfg.hidden_size) # project to llama hidden size
            )
        
class WhisperPerceptionModel(NeuralModule, Exportable):
    def __init__(self, cfg):
        super().__init__()
        logging.info(cfg)
        whisper_model = WhisperModel.from_pretrained(
            cfg.pretrained_audio_model # model_name_or_path
        )
        logging.info(f"Loaded whisper model from {cfg.pretrained_audio_model}")

        self.encoder = whisper_model.encoder

        # todo: add prompt based on cfg
        self.prompt_size = cfg.perception.prompt_size
        
        self.modality_adapter = PromptAdapter(cfg, whisper_config=whisper_model.config)


    def forward(self, input_signal, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        audio_features = self.forward_whisper(input_features=input_signal)
        
        audio_feat_lens = torch.zeros(input_signal.size(0)).long().to(input_signal.device) # for original SALM
        
        return audio_features, audio_feat_lens


    def forward_whisper(self, input_features, attention_mask=None, head_mask=None, output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        """
          @kehan modified from whisper encoder
        """
        bs = input_features.size(0)
        
        expected_seq_length = self.encoder.config.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        

        inputs_embeds = nn.functional.gelu(self.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.encoder.embed_positions.weight[:self.encoder.config.max_source_positions, :] # @kehan

        hidden_states = inputs_embeds + embed_pos
        # hidden_states = nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        
        
        features_length = hidden_states.size(1)
        layer_prompt_outputs = []
        for idx, encoder_layer in enumerate(self.encoder.layers):
            layer_prompt = self.modality_adapter.layer_prompts[idx].weight.unsqueeze(0).repeat(bs, 1, 1)
            
            # audio output
            with torch.no_grad():
                layer_output = self.forward_encoder_layer(encoder_layer=encoder_layer,hidden_states=hidden_states)

            layer_prompt_output = self.forward_encoder_layer(encoder_layer=encoder_layer, prompts=layer_prompt, hidden_states=hidden_states) # Q: prompts K,V: prompts+hidden_state
            layer_prompt_output = layer_prompt_output[0]

            hidden_states = layer_output[0]

            assert layer_prompt_output.size(1) == self.prompt_size
            assert hidden_states.size(1) == features_length

            layer_prompt_outputs.append(layer_prompt_output)

        # Other implementation
        
        # (b, 1, tgt_len, src_len)
        # total_length = self.prompt_size + features_length
        # attention_mask = torch.ones(total_length, total_length)
        # attention_mask[:self.prompt_size, :] = 0
        # attention_mask[self.prompt_size:, self.prompt_size:] = 0
        # attention_mask = attention_mask * -1e9
        # attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1) # (b, 1, total_length, total_length)
        # attention_mask = attention_mask.to(input_features.device)
        # layer_prompt_outputs = []
        # for idx, encoder_layer in enumerate(self.encoder.layers):
        #     layer_prompt = self.layer_prompts[idx].weight.unsqueeze(0).repeat(bs, 1, 1) # (b, prompt_size, d_model)
        #     hidden_states = torch.cat([layer_prompt, hidden_states], dim=1) # (b, prompt_size + features_length, d_model)
        #     layer_outputs = encoder_layer(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         layer_head_mask=None,
        #         output_attentions=None,
        #     )
        #     # layer_outputs[0]: (bs, prompt_size+feature_length, d_model)
        #     layer_prompt_output = layer_outputs[0][:, :self.prompt_size, :] # (b, prompt_size)
        #     hidden_states = layer_outputs[0][:, self.prompt_size:, :] 

        #     assert layer_prompt_output.size(1) == self.prompt_size
        #     assert hidden_states.size(1) == features_length

        #     layer_prompt_outputs.append(layer_prompt_output)

        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)

        norm_weights = torch.nn.functional.softmax(self.modality_adapter.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)

        assert all(abs(norm_weights.sum(1)- 1.0) < 0.0001)

        prompt_output = (layer_prompt_outputs * norm_weights).sum(dim=2) # (b, prompt_size, d_model)
        assert prompt_output.size(1) == self.prompt_size

        prompt_output = self.modality_adapter.proj(prompt_output) # (b, prompt_size, hidden_size)

        return prompt_output

    def forward_encoder_layer(
        self,
        encoder_layer,
        hidden_states,
        prompts=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        
        if prompts is not None:
            # query: prompts
            # key,value: prompts + hidden_states
            kev_value_state = torch.cat([prompts, hidden_states], dim=1)
            hidden_states = prompts

            residual = hidden_states
            hidden_states = encoder_layer.self_attn_layer_norm(hidden_states)

            hidden_states, attn_weights, _ = encoder_layer.self_attn(
                hidden_states=hidden_states,
                key_value_states=kev_value_state, # @kehan
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
        else:
            # original attention
            residual = hidden_states
            hidden_states = encoder_layer.self_attn_layer_norm(hidden_states)
            hidden_states, attn_weights, _ = encoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        hidden_states = nn.functional.dropout(hidden_states, p=encoder_layer.dropout, training=encoder_layer.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = encoder_layer.final_layer_norm(hidden_states)
        hidden_states = encoder_layer.activation_fn(encoder_layer.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=encoder_layer.activation_dropout, training=encoder_layer.training)
        hidden_states = encoder_layer.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=encoder_layer.dropout, training=encoder_layer.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs