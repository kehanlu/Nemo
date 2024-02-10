import itertools
import json
import os
from typing import Dict, List, Optional, Union

import sacrebleu
import torch
from hydra.utils import get_class
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.models import ASRModel, EncDecSpeakerLabelModel, SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.multimodal.speechllm.data.audio_text_qa_dataset import (
    get_aqa_dataset_from_config,
    get_tarred_aqa_dataset_from_config,
)
from nemo.collections.multimodal.speechllm.modules.common.audio_text_generation_utils import generate
from nemo.collections.multimodal.speechllm.modules.speechllm_perception import (
    AudioPerceptionModel,
    MultiAudioPerceptionModel,
)
from nemo.collections.multimodal.speechllm.parts.utils.data_utils import remove_text_pc, to_cuda
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTLoRAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_utils import get_computeprob_response
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector, PEFTSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes import ModelPT
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import InferenceParams, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


from nemo.collections.multimodal.speechllm.models.speechllm_models import ModularAudioGPTLoRAModel

__all__ = ["WhisperLlamaModel"]


# @kehan
from nemo.collections.multimodal.speechllm.data.whisper_llama_qa_dataset import (
    get_whisper_llama_dataset_from_config
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

from nemo.collections.multimodal.speechllm.modules.whisper_perception import WhisperPerceptionModel


class WhisperLlamaModel(ModularAudioGPTLoRAModel):
    def __init__(self, cfg, trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        self.perception = WhisperPerceptionModel(cfg)

        self.setup_optimizer_param_groups()
        self.configure_optimizers()
        self.summarize(max_depth=2)


        self.prompt_size = cfg.perception.prompt_size
        logging.info(self)


    def forward(self, audio_batch, checkpoint_activations_all_layers):

        input_ids, input_length, labels, loss_mask, audio_signal = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
            audio_batch['audio_signal'],
        )

        input_embeddings, attention_mask, labels, loss_mask, _ = self.prepare_llm_input(audio_batch)
        
        if self.mcore_gpt:
            output = self.model(
                input_ids=None,
                position_ids=None,
                decoder_input=input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            output = self.model(
                input_ids=None,
                position_ids=None,
                encoder_input=input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )
        return output, loss_mask

    
    def prepare_llm_input(self, audio_batch):
        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )

        # Whisper encoder output (Prompt)
        audio_features, audio_feat_lens = self.perception(
            input_signal=input_signal,
        )

        # inject audio features into LLM input_embeddings        
        input_embeddings, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            audio_features=audio_features, input_ids=input_ids, 
        )

        return input_embeddings, attention_mask, labels, loss_mask, encoder_length
        
    
    
    def inject_perception_input(self, 
                                audio_features,
                                audio_feature_lengths=None, 
                                input_ids=None,
                                input_lengths=None,
                                context_start_idx=None
                                ):
        # replace input_ids with audio_features
        # the input_ids is pad with [-42]*prompt_size to indicate the audio prompt
        # and 0 is the index of unk token
        audio_start_ids = input_ids.argmin(dim=-1) # A trick to detect start of audio start
        
        for idx, audio_start_id in enumerate(audio_start_ids):
            input_ids[idx, audio_start_id:audio_start_id+self.prompt_size] = 2 # replace -42 with valid token(<unk>)
        

        lm_embedding = (
            self.model.language_model.embedding if hasattr(self.model, 'language_model') else self.model.embedding
        )
        modified_embeddings = []
        for idx, audio_start_id in enumerate(audio_start_ids):

            before_audio = lm_embedding.word_embeddings(input_ids[idx, :audio_start_id].unsqueeze(0)).squeeze(0)
            after_audio = lm_embedding.word_embeddings(input_ids[idx, audio_start_id+self.prompt_size:].unsqueeze(0)).squeeze(0)
            current_audio_features = audio_features[idx, :]
            

            modified_embedding = torch.cat([before_audio, current_audio_features, after_audio], dim=0)
            modified_embeddings.append(modified_embedding)

        input_embeddings = torch.stack(modified_embeddings, dim=0).contiguous()

        attention_mask = self._create_attention_mask(input_embeddings)
        position_ids = build_position_ids(input_embeddings[:, :, 0])
        
        # Add position embeddings
        if (
            getattr(lm_embedding, "position_embeddings", None) is not None
            and lm_embedding.position_embedding_type == 'learned_absolute'
        ):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            input_embeddings = input_embeddings + position_embeddings
        else:
            input_embeddings = input_embeddings
        encoder_max_length = input_embeddings.shape[1]
        if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
            input_embeddings = input_embeddings.transpose(0, 1).contiguous()
        if self.cfg.get("sequence_parallel", False):
            input_embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(input_embeddings)

        # encoder_length = 0 # dummy
        encoder_length = torch.zeros(input_embeddings.size(0)).long().to(input_embeddings.device)
        return input_embeddings, attention_mask, encoder_length, position_ids, encoder_max_length

    def setup_optimizer_param_groups(self):
        # setp up which part of model should update
        self.unfreeze()
        known_groups = []
        if self.cfg.get('freeze_llm', True):
            for param in self.model.parameters():
                param.requires_grad = False
            known_groups.append('model.')
        
        if self.cfg.get('freeze_audio_encoder', True):
            # self.perception.encoder.freeze()
            for param in self.perception.encoder.parameters():
                param.requires_grad = False

            known_groups.append('perception.encoder.')
        
        if self.cfg.get('freeze_layer_prompts', False):
            # self.perception.layer_prompts.freeze()
            for param in self.perception.modality_adapter.layer_prompts.parameters():
                param.requires_grad = False
            known_groups.append('perception.modality_adapter.')


        opt_params = []
        if self.cfg.get('freeze_lora', False):
            logging.warning("LoRA adapter is freezed.")
        else:
            for _, module in self.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                    module.set_enabled_adapters(enabled=True)
                    module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                    opt_params += [p for p in module.parameters()]

        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(f"{group}.")
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

        for n, p in self.named_parameters():
            is_unknown = True
            for group in known_groups:
                if n.startswith(group):
                    is_unknown = False
            if is_unknown:
                opt_params.append(p)
                logging.info(f"Trainable: {n}")

        param_groups = [{"params": opt_params}] + param_groups

        self._optimizer_param_groups = param_groups
        logging.info(f"Optimizer groups set:\n{self.summarize(max_depth=3)}")

    def state_dict(self):
        if self.setup_complete:
            # Once setup is complete we only need adapter and perception model.
            if self.cfg.get('freeze_llm', True):
                return_state_dict = self.get_peft_state_dict()
            else:
                return_state_dict = self.model.state_dict(prefix="model.")
            # kehan: only layer_prompts
            state_dict = self.perception.modality_adapter.state_dict(prefix="perception.modality_adapter.")
            return_state_dict.update(state_dict)
            logging.info(self.perception.modality_adapter.layer_weights.softmax(-1))
            return return_state_dict
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix="model.")


    def _modify_audio_encoder_config(self, gpt_cfg, audio_cfg, speaker_cfg=None):
        pass
    
    @classmethod
    def _modify_config(cls, gpt_cfg, cfg):
        """
        kehan: discard audio_cfg
        """
        OmegaConf.set_struct(gpt_cfg, True)
        OmegaConf.resolve(cfg)
        with open_dict(gpt_cfg):
            gpt_cfg.freeze_llm = cfg.model.get('freeze_llm', True)
            gpt_cfg.freeze_audio_encoder = cfg.model.get('freeze_audio_encoder', False)
            gpt_cfg.freeze_modality_adapter = cfg.model.get('freeze_modality_adapter', False)
            gpt_cfg.freeze_lora = cfg.model.get('freeze_lora', False) # @kehan
            gpt_cfg.freeze_layer_prompts = cfg.model.get('freeze_layer_prompts', False) # @kehan

            gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
            gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
            gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
            gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
            gpt_cfg.tensor_model_parallel_size = cfg.model.get(
                "tensor_model_parallel_size", gpt_cfg.tensor_model_parallel_size
            )
            gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
            gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
            gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
            gpt_cfg.data = cfg.model.data
            gpt_cfg.optim = cfg.model.optim
            gpt_cfg.precision = cfg.trainer.precision
            gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
            gpt_cfg.restore_from_path = cfg.model.restore_from_path
            gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
            gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
            gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
            gpt_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
            gpt_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
            gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
            gpt_cfg.peft = cfg.model.peft
            # for AudioGPTLoRAModel
            gpt_cfg.target = f"{cls.__module__}.{cls.__name__}"
            gpt_cfg.perception = cfg.model.perception

            # @kehan
            # cls._modify_audio_encoder_config(gpt_cfg, audio_cfg, speaker_cfg)
            gpt_cfg.pretrained_audio_model = cfg.model.pretrained_audio_model

            override_vocab_size = cfg.model.get('override_vocab_size', None)
            if override_vocab_size is not None:
                gpt_cfg.override_vocab_size = override_vocab_size
            # This is needed when modifying a hparam file directly to load `.ckpt` files.
            # This is not needed to modify the cfg in `.nemo` files.
            # if add_cfg_to_tree:
            #     OmegaConf.resolve(gpt_cfg)
            #     gpt_cfg.cfg = gpt_cfg

        return gpt_cfg

    @classmethod
    def restore_from_pretrained_models(
        cls, cfg: Optional[Union[OmegaConf, str]] = None, trainer: Optional[Trainer] = None,
    ):
        if (
            cfg.model.get("pretrained_audio_model", None) is None
            and cfg.model.perception.get("encoders", None) is None
        ):
            raise RuntimeError("PEFT training needs at least one pretrained audio model present.")

        if not cfg.model.restore_from_path:
            raise RuntimeError("PEFT training needs a trained base model present.")

        base_model_save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.restore_from_path):
            base_model_save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
        
        # llama
        base_model_cfg = cls.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=base_model_save_restore_connector,
        )

        # audio_model, audio_model_cfg = cls.get_audio_encoder_models_and_configs(cfg)
        # speaker_model, speaker_cfg = cls.get_speaker_model_and_config(cfg)
        model_cfg = cls._modify_config(gpt_cfg=base_model_cfg, cfg=cfg)

        save_restore_connector = PEFTSaveRestoreConnector(
            peft_model_nemo_path=cfg.model.peft.restore_from_path,
            peft_model_ckpt_path=cfg.model.peft.restore_from_path,
        )

        if os.path.isdir(cfg.model.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

        # load llm
        model = cls.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=model_cfg,
            save_restore_connector=save_restore_connector,
            strict=False,
        )
        logging.info("="*60)
        logging.info(model)

        # load audio model weights
        # @kehan: handle in perception model
        # model = cls._load_pretrained_audio_weights(cfg, model, audio_model, speaker_model)

        if 'inference' in cfg:
            inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
            model.set_inference_config(inference_cfg)
        return model

    def get_audio_encoder_models_and_configs(cfg):
        # @kehan we don't need this
        # load whisper in perception model
        pass



    def _build_dataset(self, data_cfg, is_train=True):
        logging.info(data_cfg)
        if 'augmentor' in data_cfg:
            augmentor = process_augmentations(
                data_cfg['augmentor'], global_rank=self.global_rank, world_size=self.world_size
            )
        else:
            augmentor = None

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings


        # kehan:
        if data_cfg.get('is_tarred', False):
            return get_tarred_aqa_dataset_from_config(
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                sep_id=self.sep_id,
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                virtual_tokens=self.virtual_tokens,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            return get_whisper_llama_dataset_from_config(
                manifest_filepath=data_cfg.manifest_filepath,
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                is_train=is_train,
                sep_id=self.sep_id,
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                virtual_tokens=self.virtual_tokens,
            )
        
        