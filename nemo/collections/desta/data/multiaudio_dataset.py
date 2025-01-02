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

def find_all_indices(string_list, target):
    return [index for index, item in enumerate(string_list) if item == target]
class MultiAudioSpeechLlamaDataset(SpeechLlamaDataset):
    """
    Support multiple audio inputs.


    The dataset is a jsonl file with the following format:
    {
        "id": "VCTK-Corpus/wav48/p292/p292_150.wav",
        "audios": [{
            "audio_filepath": "VCTK-Corpus/wav48/p292/p292_150.wav",
            "duration": 3.2049583333333334,
            "transcription": "They missed the boat and everyone suffers."
        }, {
            "audio_filepath": "VCTK-Corpus/wav48/p292/p292_150.wav",
            "duration": 3.2049583333333334,
            "transcription": "They missed the boat and everyone suffers."
        }],
        "messages": [{
                "role": "user",
                "content": "What can you hear from the audio?"
            },{
                "role": "assistant",
                "content": "From the audio, I can hear:\n\n* A male voice with a Northern Irish accent\n* A statement: \"They missed the boat and everyone suffers.\"\n* The speaker has a neutral emotion, with no apparent excitement, stress, or enthusiasm\n* The audio is 3 seconds long"
            }],
        "target": string,
    }


    Compare to /NeMo/nemo/collections/desta/data/simple_dataset.py:
    - change the input jsonl format
        - pack audios into a list
    - instead writing chat template in the config, directly use "messages" in each input
    - change the processing logic in finding audio position
    - add audio_locator as a special token

    """

    def __init__(self, cfg, data_cfg):
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.cfg.dataset.audio_locator]})
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.speech_encoder.model_id)

        
        logging.info(self.data_cfg)

        self.dataset = datasets.load_dataset(
            "json", data_files=self.data_cfg.manifest_filepaths
        )["train"]
        
        logging.info(f"Loaded dataset from {self.data_cfg.manifest_filepaths}")
        logging.info(f"Number of Lines: {len(self.dataset)}")
        logging.info(f"Audio Locator: {self.cfg.dataset.audio_locator}")
        logging.info(self.dataset)

        self.dataset = self.dataset.map(
            self.batchified_preprocess_function,
            batched=True,
            batch_size=64,
        )

    def batchified_preprocess_function(self, examples):
        """
        audio_locator: self.cfg.dataset.audio_locator

        input(jsonl): 
            {
                "audios": [{duration, transcription, audio_filepath}],
                "messages": string,
                "target": string
            }

        output:
            {
                "audios": [{duration, transcription, absolute audio_filepath, position}]
                "context": string,
                "target": string,
            }
        
        """

        formatted_contexts = []
        for messages, audios in zip(examples["messages"], examples["audios"]):
            context = self.tokenizer.apply_chat_template(messages ,tokenize=False, add_generation_prompt=True)
            for audio in audios:
                transcription = audio["transcription"]
                context = context.replace("{transcription}", transcription, 1) # context = context with transcription

            # find audio position from tokenized context list
            # [A, B, C, audio_locator, D, E, F, audio_locator, G, H]
            audio_positions = find_all_indices(self.tokenizer.tokenize(context), self.cfg.dataset.audio_locator)
            assert len(audio_positions) == len(audios), f"Length mismatch: {audio_positions} and {audios}"
            # [A, B, C, D, E, F, G, H]
            context = context.replace(self.cfg.dataset.audio_locator, "")
            

            audio_position_shift = 0 # since we remove audio_locator from context, the actual position is shifted in the tokenized context list
            for audio in audios:
                audio["audio_filepath"] = str(self.data_cfg.data_root) + "/" + audio["audio_filepath"]
                audio["position"] = audio_positions.pop(0) - audio_position_shift
                
                audio_position_shift += 1

            formatted_contexts.append(context)
        
        # target
        for target in examples["target"]:
            target = target + self.tokenizer.eos_token
            
        examples["context"] = formatted_contexts

        return examples

    
    def collate_fn(self, batch):

        text_inputs = self.tokenizer([item['context']+item["target"] for item in batch], truncation=True, padding="longest", max_length=1024, return_tensors="pt", return_length=True, add_special_tokens=False)
        context_inputs = self.tokenizer(
            [item['context'] for item in batch], truncation=True, padding="longest", max_length=1024, return_tensors="pt", return_length=True, add_special_tokens=False
        )
        
        labels = torch.full_like(text_inputs['input_ids'], -100)
        features = []
        audios = []

        for i, item in enumerate(batch):
            # build labels (padding_side = left)
            total_length = text_inputs['length'][i]
            context_length = len(self.tokenizer.tokenize(item["context"]))
            
            pad_length = total_length - text_inputs["attention_mask"][i].sum()

            start_answer_position = pad_length + context_length
            labels[i, start_answer_position:] = text_inputs['input_ids'][i, start_answer_position:]

            item_audios = []
            for audio in item["audios"]:
                feature = AudioSegment.from_file(
                    audio["audio_filepath"],
                        target_sr=16000,
                        duration=audio["duration"],
                    ).samples
                    
                feature = self.processor(feature, sampling_rate=16000, return_tensors="pt").input_features[0]
                features.append(feature)
                
                item_audios.append({
                    "position": audio["position"] + pad_length,
                    "feature_index": len(features) - 1,
                })
            audios.append(item_audios)
        logging.info(f"audios: {audios}")
                
        features = torch.stack(features)
        
        # we use labels for calculating loss
        # "target" is the dataset key name
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'audio_features': features,
            "audios": audios,

            'context_input_ids': context_inputs['input_ids'],
            'context_attention_mask': context_inputs['attention_mask'],

            # for debugging
            'contexts': [item['context'] for item in batch],
            'targets': [item['target'] for item in batch],
            
            "metadata": [item for item in batch],
        }