import json
from typing import Any, Dict, Iterable, List, Optional, Union

import torch 

from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

from transformers import AutoTokenizer, WhisperProcessor

__all__ = [
    "SpeechLlamaDataset",
]

class SpeechLlamaCollection():
    """
    load manifest to a list.
    """
    def __init__(self, 
                 manifests_files,
                 max_duration=10e6,
                 min_duration=0):
        
        data = []
        total_duration = 0
        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            if item['duration'] > max_duration:
                continue
            elif item['duration'] < min_duration:
                continue
            else:
                total_duration += item['duration']

            data.append(item)


        logging.info(f"Dataset loaded with {len(data)} files totalling {total_duration:.2f} hours")
        
        self.data = data
    
    def __parse_item(self, line, manifest_file) -> Dict[str, Any]:
        item = json.loads(line)

        assert item.get("audio_file") or item.get("audio_filepath")
        assert item.get("duration")
        assert item.get("answer")

        
        # Map audio to actual filepath
        if 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        item['audio_file'] = [item['audio_file']] if isinstance(item["audio_file"], str) else item['audio_file']
        
        # handle multiple audio
        item['audio_file'] = [
            manifest.get_full_path(audio_file=filepath, manifest_file=manifest_file)
            for filepath in item['audio_file']
        ]
        
        return item
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)



class SpeechLlamaDataset(Dataset):
    
    def __init__(self,
                 cfg,
                 data_cfg,
        ):
        """
        config
            manifest_filepath: str or list
        """
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg # dataset dependent configuration
        logging.info(self.data_cfg)

        # Load manifest
        # if we have many manifest files, we will combine them all into the one collection
        manifest_filepaths = self.data_cfg.manifest_filepaths if isinstance(self.data_cfg.manifest_filepaths, list) else [self.data_cfg.manifest_filepaths]
        self.collection = SpeechLlamaCollection(
            manifests_files=manifest_filepaths,
            max_duration=self.data_cfg.max_duration,
            min_duration=self.data_cfg.min_duration,
        ) 

        self.featurizer = WaveformFeaturizer(sample_rate=16000)
        
        # huggingface models
        self.whisper_processor = WhisperProcessor.from_pretrained(self.cfg.model.speech_encoder.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        
    def __len__(self):
        return len(self.collection)
    
    def __getitem__(self, idx):
        # a sample from manifest
        sample = self.collection[idx] 

                
        # load audio features
        audio_features = self._process_audio(sample)

        # apply chat template and format input_ids
        text_features = self._process_text(sample)

        return {**audio_features, **text_features} # merge 2 dictionaries
        
    
    def _process_audio(self, sample):
        feature = self.featurizer.process(
            sample["audio_file"],
            duration=sample["duration"],
        )
        # feature = self.whisper_processor(
        #         feature, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        
        return {
            "audio_feature": feature,
            # "audio_length": feature.size(0),
        }
    
    def _process_text(self, sample):
        chat_template = self.data_cfg.chat_template # {"role": "system", "content": "{system_prompt}"}
        
        text = self.tokenizer.apply_chat_template(
            chat_template,
            tokenize=False,
        ) # all text (user + model output)

        user_text = self.tokenizer.apply_chat_template(
            chat_template[:-1],
            tokenize=False,
        ) # only user input

        
        # Fill in the template
        text = self._apply_template(text, sample)
        user_text = self._apply_template(user_text, sample)
        answer_text = text[len(user_text):]
        answer_ids = self.tokenizer.encode(answer_text)


        # Process audio locator
        if hasattr(self.data_cfg, "audio_locator"):
            if len(user_text.split(self.data_cfg.audio_locator)) != 2:
                logging.warning(f"Audio locator '{self.data_cfg.audio_locator}' not found in '{user_text}'")
            
            left = user_text.split(self.data_cfg.audio_locator)[0]
            right = user_text.split(self.data_cfg.audio_locator)[1]
            
            user_text_ids = []
            user_text_ids.extend(self.tokenizer.encode(left))
            user_text_ids.extend([-42]*self.cfg.model.prompt_size) # @kehan prompt placeholder
            user_text_ids.extend(self.tokenizer.encode(right))

        input_ids = user_text_ids + answer_ids

        return {
            "input_ids": input_ids,
            "user_text_ids": user_text_ids, # helper
            "answer_ids": answer_ids, # helper
        }
        
    def _apply_template(self, text, sample):
        """
        fill the data into placeholders in the template
        """
        if hasattr(self.data_cfg, "system_prompt"):
            text = text.replace("{system_prompt}", self.data_cfg.system_prompt)
        
        text = text.replace("{transcription}", sample.get("transcription", ""))
        text = text.replace("{dataset}", sample.get("dataset", ""))
        
        text = text.replace("{question}", sample["question"])
        text = text.replace("{answer}", sample["answer"])
        
        
        return text

        
    def collate_fn(self, batch):
        # Speech encoder
        audio_features = [x["audio_features"] for x in batch]
        audio_lengths = [x["audio_length"] for x in batch]
        

        # Language model
        input_ids = [x["input_ids"][1:] for x in batch]
        labels = [x["input_ids"][:-1] for x in batch]
        
        user_text_ids = [x["user_text_ids"] for x in batch] # helper
        answer_ids = [x["answer_ids"] for x in batch] # helper
                
        