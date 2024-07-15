from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel, SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
# from nemo.collections.multimodal.speechllm.parts.utils.data_utils import align_feat_seq_list, get_nested_dict_value
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

# @kehan
from transformers import WhisperModel, BertConfig, WhisperForConditionalGeneration
from transformers.models.bert.modeling_bert import BertEncoder


class QformerConnector(NeuralModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        
        if self.cfg.model.speech_encoder.model_id == "openai/whisper-medium":
            self.target_layer_ids = [5, 11, 17, 23]
        elif self.cfg.model.speech_encoder.model_id == "openai/whisper-small":
            self.target_layer_ids = [2, 5, 8, 11]
        elif self.cfg.model.speech_encoder.model_id == "openai/whisper-tiny":
            self.target_layer_ids = [0,1,2,3]
        elif self.cfg.model.speech_encoder.model_id == "openai/whisper-large-v3":
            self.target_layer_ids = [3, 7, 11, 15, 19, 23, 27, 31]


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.cfg.model.connector.prompt_size, self.cfg.model.speech_encoder.cfg.d_model)) for _ in range(len(self.target_layer_ids))]
        )
        
        if self.cfg.model.connector.mode == "qformer_1":
            # (prompt_size, target_layers)
            self.layer_weights = nn.Parameter(torch.zeros(self.cfg.model.connector.prompt_size, len(self.target_layer_ids), dtype=torch.float))

        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = 2
        qformer_config.num_attention_heads = self.cfg.model.speech_encoder.cfg.encoder_attention_heads
        qformer_config.hidden_size = self.cfg.model.speech_encoder.cfg.d_model
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True

        self.qformer = BertEncoder(qformer_config)
        self.proj = nn.Sequential(
                nn.LayerNorm(self.cfg.model.speech_encoder.cfg.d_model),
                nn.Linear(self.cfg.model.speech_encoder.cfg.d_model, self.cfg.model.language_model.cfg.hidden_size) # project to llama hidden size
            )


class WhisperPerceptionModule(NeuralModule, Exportable):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        whisper = WhisperForConditionalGeneration.from_pretrained(self.cfg.model.speech_encoder.model_id, cache_dir="/NeMo/.cache")
        self.encoder = whisper.model.encoder

        if self.cfg.model.connector.mode == "qformer_1":
            self.connector = QformerConnector(cfg=self.cfg)
        else:
            raise NotImplementedError(f"mode {self.cfg.model.connector.mode} not implemented")

    

    def forward(self, input_features):
        bs = input_features.size(0)

        audio_features = self.forward_whisper(input_features=input_features)
        audio_feature_lengths = torch.ones(
            [bs,], dtype=torch.long, device=input_features.device 
        ) * audio_features.size(1) # assume all have same lengths
        
        
        return audio_features, audio_feature_lengths


    def forward_whisper(self, input_features):
        """
        2024.07.07 @kehan
        copy from previous implementation for qformer_1
        
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
        features_length = hidden_states.size(1)

        if self.cfg.model.connector.mode == "qformer_1" or self.cfg.model.connector.mode == "qformer_2":
            layer_prompt_outputs = []
            for idx, encoder_layer in enumerate(self.encoder.layers):
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]

                if idx in self.connector.target_layer_ids:
                    # use different prompt for different layers
                    layer_prompt = self.connector.layer_prompts[self.connector.target_layer_ids.index(idx)].expand(bs, -1, -1)
                    
                    # Qformer is a BERTEncoder(but set to decoder) from huggingface Transformers
                    qformer_output = self.connector.qformer(
                        hidden_states=layer_prompt,
                        encoder_hidden_states=hidden_states,
                    )

                    layer_prompt_output = qformer_output.last_hidden_state # (b, prompt_size, d_model)
                    layer_prompt_outputs.append(layer_prompt_output) # list of (b, prompt_size, d_model)
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)
        
        if self.cfg.model.connector.mode in ["qformer_1", "prompt_1", "prompt_2"]:
            self.norm_weights = torch.nn.functional.softmax(self.connector.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)
        else:
            raise NotImplementedError()
        

        prompt_output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
        assert prompt_output.size(1) == self.cfg.model.connector.prompt_size, prompt_output.size()

        prompt_output = self.connector.proj(prompt_output)
        
        return prompt_output