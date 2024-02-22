import copy
import io
import json
import os
from math import isclose
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.data.audio_to_text import (
    cache_datastore_manifests,
    expand_sharded_filepaths,
    shard_manifests_if_needed,
)
from nemo.collections.asr.data.audio_to_text_dataset import ConcatDataset, convert_to_config_list, get_chain_dataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.multimodal.speechllm.parts.utils.data_utils import (
    ceil_to_nearest,
    get_num_samples_from_files,
    maybe_cast_to_list,
)
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging

try:
    from megatron.core import parallel_state, tensor_parallel

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = [
    "WhisperLlamaDataset",
    "get_whisper_llama_dataset_from_config"
]

# @kehan
from nemo.collections.multimodal.speechllm.data.audio_text_qa_dataset import (
    TextProcessing, _build_loss_mask, _collate_item, _audio_text_collate_fn
)
from transformers import WhisperProcessor


class TextProcessing(object):
    """
    Text processing pipeline for AudioQuestionAnswerDataset and TarredAudioQuestionAnswerDataset.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        sample_alpha: Optional[float] = None,
        audio_locator: Optional[str] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.end_string = end_string
        self.sample_alpha = sample_alpha
        self.audio_locator = audio_locator

        if add_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if add_eos and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = self.eos_id if self.eos_id is not None else 0

        self.sep_id = sep_id if add_sep else None

        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

    def _process_example(self, context: str, output: str):
        pass



class WhisperLlamaDataset(TextProcessing, Dataset):
    """
    from MultiAudioQuestionAnswerDataset
    """
    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        index_by_file_id: bool = False,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        question_file: Optional[Union[List[str], str]] = None,
        random_context_prob: Optional[float] = None,
        random_context_num: Optional[int] = 3,
        random_context_positive_percent: Optional[float] = 0.1,
        sample_alpha: Optional[float] = None,
        audio_locator: Optional[str] = None,
        pretrained_audio_model: Optional[str] = None,
        prompt_size=32,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,
            seed=seed,
            separate_prompt_and_response_with_newline=separate_prompt_and_response_with_newline,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            prompt_template=prompt_template,
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            input_key=input_key,
            output_key=output_key,
            end_string=end_string,
            sample_alpha=sample_alpha,
            audio_locator=audio_locator,
        )

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")

        # If necessary, cache manifests and audio from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)

        # kehan: collections.ALMAudioTextCollection
        # + transcription field
        self.collection = collections.WhisperLlamaCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            max_num_samples=max_num_samples,
            question_file=question_file,
            random_context_num=random_context_num,
            random_context_positive_percent=random_context_positive_percent,
            random_context_prob=random_context_prob,
        )
        self.hf_model_id = pretrained_audio_model # openai/whisper-
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.whisper_processor = WhisperProcessor.from_pretrained(self.hf_model_id)
        self.trim = trim
        self.channel_selector = channel_selector
        self.prompt_size = prompt_size
        
        logging.info(f"audio_locator: {audio_locator}")


    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __getitem__(self, index):
        output = {"idx": index}
        sample = self.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        if sample.audio_file is not None:
            features = self.featurizer.process(
                sample.audio_file,
                offset=offset,
                duration=sample.duration,
                trim=self.trim,
                orig_sr=sample.orig_sr,
                channel_selector=self.channel_selector,
            )
            features = self.whisper_processor(
                features, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0).view(-1)

            f, fl = features, torch.tensor(features.shape[0]).long()
            output["audio_signal"] = f
            output["audio_length"] = fl
        else:
            # dummy features
            output["audio_signal"] = torch.zeros([8000])
            # accomodates normalize_batch
            output["audio_length"] = torch.tensor(8000)

        text_data = self._process_example(context=sample.question, output=sample.answer, transcription=sample.transcription)

        output.update(text_data)
        output['metadata'] = {
            'audio_filepath': sample.audio_file,
            'offset': offset,
            'duration': sample.duration,
        }
        return output

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        return _audio_text_collate_fn(
            batch=batch,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
            text_pad_id=self.pad_id,
        )

    def collate_fn(self, batch):
        return self._collate_fn(batch)

    def _process_example(self, context: str, output: str, transcription:str):
        """
        @kehan overwrite
        prompt_template=

        '<s>[INST] <<SYS>>\nHello!\n<</SYS>>\n\nWhat is the capital of France? [/INST] Paris'
        '<s>[INST] <<SYS>>\nHello!\n<</SYS>>\n\n{Q} [/INST] {A}'

        Q contains audio and transcription placeholder
        Q: xxxx [SPEECH]|audio prompt| {transcription} [/SPEECH] xxxx
        """

        if self.prompt_template is not None:
            assert f'{{{self.input_key}}}' in self.prompt_template
            assert f'{{{self.output_key}}}' in self.prompt_template
            # Make sure that '{output}' always occurs at the end of the prompt template string
            assert self.prompt_template.index(f'{{{self.output_key}}}') == len(self.prompt_template) - len(
                f'{{{self.output_key}}}'
            )

            # @kehan: Replace the transcription
            # [SPEECH]|audio| hello word[/SPEECH] What is this audio?
            context = context.replace('{transcription}', transcription)

            # Get the context by replacing only the input
            original_context = context
            context = (
                self.prompt_template.replace(f'{{{self.input_key}}}', context)
                .replace(f'{{{self.output_key}}}', '')
                .strip(' ')
            )
            # Replace the input and output placeholders with the actual input and output
            text = self.prompt_template.replace(f'{{{self.input_key}}}', original_context).replace(
                f'{{{self.output_key}}}', str(output)
            )

        elif self.separate_prompt_and_response_with_newline:
            text = context + '\n' + output
        else:
            text = context + ' ' + output

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            pre_pad = [self.tokenizer.eos_id] * self.virtual_tokens
        else:
            pre_pad = []
        answer_text = text[len(context) :]
        answer_ids = pre_pad + self.tokenizer.text_to_ids(answer_text, self.sample_alpha)
        if self.end_string:
            answer_ids += self.tokenizer.text_to_ids(self.end_string)

        if self.audio_locator is None:
            # signle audio case
            context_ids = self.tokenizer.text_to_ids(context)
            context_start_idx = [0]

        elif self.audio_locator == "|audio_prompt|":
            # @kehan: only one audio_locator
            left = context.split(self.audio_locator)[0]
            right = context.split(self.audio_locator)[1]
            
            context_ids = []
            context_ids.extend(self.tokenizer.text_to_ids(left))
            context_ids.extend([-42]*self.prompt_size) # @kehan prompt placeholder
            context_ids.extend(self.tokenizer.text_to_ids(right))
            
            context_start_idx = [0]
        else:
            # multiple audio case
            context_ids = []
            context_start_idx = []
            for context_seg in context.split(self.audio_locator):
                context_start_idx.append(len(context_ids))
                context_ids.extend(self.tokenizer.text_to_ids(context_seg))
        context_ids = pre_pad + context_ids
        context_start_idx = [x + len(pre_pad) for x in context_start_idx]

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        total_ids = len(context_ids) + max(len(answer_ids), self.tokens_to_generate)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            if self.truncation_field == "answer":
                answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            elif self.truncation_field == "context":
                context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            input_ids = input_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]
        
        # logging.info(f"input_ids: {input_ids}")

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'context_start_idx': context_start_idx,
        }

        return processed_example
    


def get_whisper_llama_dataset_from_config(
    manifest_filepath: str,
    config: DictConfig,
    tokenizer,
    augmentor,
    is_train,
    sep_id: Optional[int] = None,
    answer_only_loss: bool = True,
    virtual_tokens: int = 0,
):
    """
    @kehan get_aqa_dataset_from_config
    """
    if isinstance(config.manifest_filepath, str):
        manifest_filepath = config.manifest_filepath.split(',')
    else:
        manifest_filepath = config.manifest_filepath

    # data_cls = MultiAudioQuestionAnswerDataset if config.get('audio_locator', None) else AudioQuestionAnswerDataset
    # kehan
    data_cls = WhisperLlamaDataset
    datasets = []
    if is_train:
        # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
        # that is of the format [weight1,file_name1,weight2,file_name2,...]
        concat_sampling_probabilities = config.get('concat_sampling_probabilities', None)
        if concat_sampling_probabilities is None:
            concat_sampling_probabilities = [1.0 / len(manifest_filepath)] * len(manifest_filepath)
        elif len(config.get('concat_sampling_probabilities', None)) != len(manifest_filepath):
            raise ValueError(
                (
                    f"concat_sampling_probabilities must be of the same size as manifest_filepath.",
                    f"Provided size {len(config.concat_sampling_probabilities)}, number of datasets {len(manifest_filepath)}",
                )
            )
        data_prefix = []
        for weight, prefix in zip(concat_sampling_probabilities, manifest_filepath):
            data_prefix.append(weight)
            data_prefix.append(prefix)

        num_samples_per_dataset = get_num_samples_from_files(manifest_filepath)
        num_train_samples = [len(manifest_filepath) * max(num_samples_per_dataset)]
        _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
        num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
    else:
        num_train_samples_per_dataset = [[None]] * len(manifest_filepath)

    for dataset_idx, (file_path, num_samples) in enumerate(zip(manifest_filepath, num_train_samples_per_dataset)):
        question_file = config.get('question_file', None)
        if isinstance(question_file, ListConfig) and len(question_file) == len(manifest_filepath):
            question_file = question_file[dataset_idx]

        assert config.get("prompt_size")
        dataset = data_cls(
            manifest_filepath=file_path,
            tokenizer=tokenizer,
            sample_rate=config.sample_rate,
            int_values=config.get('int_values', False),
            augmentor=augmentor,
            max_duration=getattr(config, 'max_duration', None),
            min_duration=getattr(config, 'min_duration', None),
            max_utts=getattr(config, 'max_utts', -1),
            trim=getattr(config, 'trim_silence', False),
            channel_selector=getattr(config, 'channel_selector', None),
            max_seq_length=config.max_seq_length,
            min_seq_length=config.min_seq_length,
            add_bos=config.get('add_bos', False),
            add_eos=config.get('add_eos', True),
            add_sep=config.get('add_sep', False),
            sep_id=sep_id,
            max_num_samples=None, # kehan
            seed=config.get('seed', 1234),
            separate_prompt_and_response_with_newline=config.get('separate_prompt_and_response_with_newline', True),
            answer_only_loss=answer_only_loss,
            truncation_field=config.get('truncation_field', 'context'),
            pad_to_max_length=config.get('pad_to_max_length', False),
            prompt_template=config.get('prompt_template', None),
            virtual_tokens=virtual_tokens,
            tokens_to_generate=config.get(
                'tokens_to_generate', 0
            ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
            input_key=config.get('input_key', 'input'),
            output_key=config.get('output_key', 'output'),
            end_string=config.get('end_string', None),
            sample_alpha=config.get('sample_alpha', None),
            random_context_prob=config.get('random_context_prob', None),
            random_context_num=config.get('random_context_num', 3),
            random_context_positive_percent=config.get('random_context_positive_percent', 0.1),
            question_file=question_file,
            audio_locator=config.get('audio_locator', None),

            pretrained_audio_model=config.get("pretrained_audio_model", "openai/whisper-large-v3"),
            prompt_size=config.get("prompt_size", 32),
            # kehan
        )
        if config.get("pretrained_audio_model") is None:
            logging.warning("No pretrained audio model is set.")
        datasets.append(dataset)

    if is_train:
        dataset = BlendableDataset(
            datasets=datasets, weights=concat_sampling_probabilities, size=num_train_samples_after_blend
        )
        return dataset
    else:
        return datasets