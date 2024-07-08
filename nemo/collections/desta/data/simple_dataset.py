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
from khlu_utils import print_c

class SpeechLlamaDataset():
    def __init__(self, cfg, data_cfg):
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

        
        logging.info(self.data_cfg)

        self.dataset = datasets.load_dataset(
            "json", data_files=self.data_cfg.manifest_filepaths
        )["train"]
        
        logging.info(f"Loaded dataset from {self.data_cfg.manifest_filepaths}")
        logging.info(f"Number of files: {len(self.dataset)}")
        logging.info(f"Duration: {sum(self.dataset['duration']) / 3600:.2f}")
        logging.info(self.dataset)

        # self.dataset = self.dataset.map(
        #     self.preprocess_function
        # )
        self.dataset = self.dataset.map(
            self.batchified_preprocess_function,
            batched=True,
            batch_size=256,
        )

    def preprocess_function(self, example):
        example["audio_filepath"] = str(self.data_cfg.data_root) + "/" + example["audio_filepath"] 

        context = self.tokenizer.apply_chat_template(self.data_cfg.chat_template, tokenize=False, add_generation_prompt=True) # all before answer

        context = context.format(**{key: example[key] for key in self.data_cfg.replace_keys})
        assert len(context.split(self.cfg.dataset.audio_locator)) == 2, f"Audio locator {self.cfg.dataset.audio_locator} not found in inputs: {context}"

        left_text, right_text = context.split(self.cfg.dataset.audio_locator)
        example["audio_position"] = len(self.tokenizer.tokenize(left_text))

        example["context"] = left_text + right_text
        example["answer"] = example["answer"] + self.tokenizer.eos_token
        return example
    
    def batchified_preprocess_function(self, examples):
        # Add data root to audio filepaths
        examples["audio_filepath"] = [str(self.data_cfg.data_root) + "/" + filepath for filepath in examples["audio_filepath"]]

        # Apply chat template to create contexts
        context = self.tokenizer.apply_chat_template(
            self.data_cfg.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format contexts with example data
        formatted_contexts = []
        for i in range(len(examples["audio_filepath"])):
            example_dict = {key: examples[key][i] for key in self.data_cfg.replace_keys}
            formatted_contexts.append(context.format(**example_dict))

        # Process each context to find audio position and split text
        audio_positions = []
        left_texts = []
        right_texts = []
        for context in formatted_contexts:
            assert len(context.split(self.cfg.dataset.audio_locator)) == 2, f"Audio locator {self.cfg.dataset.audio_locator} not found in inputs: {context}"
            left_text, right_text = context.split(self.cfg.dataset.audio_locator)
            audio_positions.append(len(self.tokenizer.tokenize(left_text)))
            left_texts.append(left_text)
            right_texts.append(right_text)

        # Update examples with new data
        examples["audio_position"] = audio_positions
        examples["context"] = [left + right for left, right in zip(left_texts, right_texts)]
        examples["answer"] = [answer + self.tokenizer.eos_token for answer in examples["answer"]]

        return examples
    
    def collate_fn(self, batch):
        text_inputs = self.tokenizer([item['context']+item["answer"] for item in batch], truncation=True, padding="longest", max_length=200, return_tensors="pt", return_length=True, add_special_tokens=False)
        context_inputs = self.tokenizer(
            [item['context'] for item in batch], truncation=True, padding="longest", max_length=200, return_tensors="pt", return_length=True, add_special_tokens=False
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
            audio_positions.append(item["audio_position"] + pad_length) # padding left

            # load audios
            feature = AudioSegment.from_file(
                item["audio_filepath"],
                target_sr=16000,
                duration=item["duration"],
            ).samples
            features.append(feature)

        features = self.processor(features, sampling_rate=16000, return_tensors="pt").input_features
        
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'audio_features': features,
            'audio_positions': torch.stack(audio_positions),

            'context_input_ids': context_inputs['input_ids'],
            'context_attention_mask': context_inputs['attention_mask'],

            # for debugging
            'context': [item['context'] for item in batch],
            'answer': [item['answer'] for item in batch],
            
            "metadata": [item for item in batch],
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]