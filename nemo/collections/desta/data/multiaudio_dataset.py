import json
from typing import Any, Dict, Iterable, List, Optional, Union

import torch 

from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

from transformers import AutoTokenizer, WhisperProcessor, AutoProcessor
import datasets

from nemo.collections.desta.data.simple_dataset import SpeechLlamaDataset

class MultiAudioSpeechLlamaDataset(SpeechLlamaDataset):
    """
    Support multiple audio inputs.
    """

    def __init__(self, cfg, data_cfg):
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.processor = AutoProcessor.from_pretrained("openai/whisper-small")

        
        logging.info(self.data_cfg)

        self.dataset = datasets.load_dataset(
            "json", data_files=self.data_cfg.manifest_filepaths
        )["train"]
        
        logging.info(f"Loaded dataset from {self.data_cfg.manifest_filepaths}")
        logging.info(f"Number of files: {len(self.dataset)}")
        # logging.info(f"Duration: {sum(self.dataset['duration']) / 3600:.2f}")
        logging.info(self.dataset)

        # self.dataset = self.dataset.map(
        #     self.preprocess_function
        # )
        self.dataset = self.dataset.map(
            self.preprocess_function,
        )

    def preprocess_function(self, example):
        assert len(example["audio_filepath"]) == len(example["transcription"]) == len(example["duration"]), f"Length mismatch: {example}"

        example["audio_filepath"] = [
            str(self.data_cfg.data_root) + "/" + item for item in example["audio_filepath"]
        ] # list of filepaths

        context = self.tokenizer.apply_chat_template(self.data_cfg.chat_template, tokenize=False, add_generation_prompt=True) # all before answer

        for transcription in example["transcription"]:
            context = context.replace("{transcription}", transcription, 1)
        
        context = context.format(**{key: example[key] for key in self.data_cfg.replace_keys}) # input, input1, input2...
        
        
        self.tokenizer.add_tokens([self.cfg.dataset.audio_locator])

        audio_positions = []
        new_context = []
        for i, token in enumerate(self.tokenizer.tokenize(context)):
            if token == "|audio_features|":
                audio_positions.append(len(new_context))
            else:
                new_context.append(token)
        
        new_context = self.tokenizer.convert_tokens_to_string(new_context)

        example["audio_position"] = audio_positions

        example["context"] = new_context
        example["target"] = example["target"] + self.tokenizer.eos_token
        return example

    
    def collate_fn(self, batch):

        text_inputs = self.tokenizer([item['context']+item["target"] for item in batch], truncation=True, padding="longest", max_length=1024, return_tensors="pt", return_length=True, add_special_tokens=False)
        context_inputs = self.tokenizer(
            [item['context'] for item in batch], truncation=True, padding="longest", max_length=1024, return_tensors="pt", return_length=True, add_special_tokens=False
        )
        
        labels = torch.full_like(text_inputs['input_ids'], -100)
        features = []
        audio_positions = []
        for i, item in enumerate(batch):
            # build labels (padding_side = left)
            total_length = text_inputs['length'][i]
            context_length = len(self.tokenizer.tokenize(item["context"]))
            
            pad_length = total_length - text_inputs["attention_mask"][i].sum()

            start_answer_position = pad_length + context_length
            labels[i, start_answer_position:] = text_inputs['input_ids'][i, start_answer_position:]

            # audio position
            audio_positions.append(torch.tensor(item["audio_position"]) + pad_length) # padding left

            item_features = []
            for audio_filepath, duration in zip(item["audio_filepath"], item["duration"]):
                feature = AudioSegment.from_file(
                    audio_filepath,
                    target_sr=16000,
                    duration=duration,
                ).samples
                item_features.append(feature)
            
            item_features = self.processor(item_features, sampling_rate=16000, return_tensors="pt").input_features
            features.append(item_features)
        
        features = torch.stack(features)
        
        # we use labels for calculating loss
        # "target" is the dataset key name
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'audio_features': features,

            'audio_positions': torch.stack(audio_positions),

            'context_input_ids': context_inputs['input_ids'],
            'context_attention_mask': context_inputs['attention_mask'],

            # for debugging
            'contexts': [item['context'] for item in batch],
            'targets': [item['target'] for item in batch],
            
            "metadata": [item for item in batch],
        }