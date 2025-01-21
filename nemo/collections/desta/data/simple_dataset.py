import json
from typing import Any, Dict, Iterable, List, Optional, Union

import torch 

from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

from transformers import AutoTokenizer, WhisperProcessor, AutoProcessor, AutoFeatureExtractor
import datasets
import os

class SpeechLlamaDataset():
    def __init__(self, cfg, data_cfg):
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.cfg.model.language_model.model_id == "meta-llama/Llama-3.1-8B-Instruct":
            # overwrite chat_template that remove cutting knowledge date
            self.tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

        self.processor = AutoFeatureExtractor.from_pretrained(self.cfg.model.speech_encoder.model_id)

        
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
            batch_size=32,
        )

    def preprocess_function(self, example):
        example["audio_filepath"] = str(self.data_cfg.data_root) + "/" + example["audio_filepath"] 

        context = self.tokenizer.apply_chat_template(self.data_cfg.chat_template, tokenize=False, add_generation_prompt=True) # all before answer

        context = context.format(**{key: example[key] for key in self.data_cfg.replace_keys})
        assert len(context.split(self.cfg.dataset.audio_locator)) == 2, f"Audio locator {self.cfg.dataset.audio_locator} not found in inputs: {context}"

        left_text, right_text = context.split(self.cfg.dataset.audio_locator)
        example["audio_position"] = len(self.tokenizer.tokenize(left_text))

        example["context"] = left_text + right_text
        example["target"] = example["target"] + self.tokenizer.eos_token
        return example
    
    def batchified_preprocess_function(self, examples):
        # Add data root to audio filepaths
        examples["audio_filepath"] = [str(self.data_cfg.data_root) + "/" + filepath for filepath in examples["audio_filepath"]]

        # check file exists
        for filepath in examples["audio_filepath"]:
            assert os.path.exists(filepath), f"File not found: {filepath}"

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
        examples["target"] = [target + self.tokenizer.eos_token for target in examples["target"]]

        return examples
    
    def collate_fn(self, batch):
        text_inputs = self.tokenizer([item['context']+item["target"] for item in batch], truncation=True, padding="longest", max_length=200, return_tensors="pt", return_length=True, add_special_tokens=False)
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
                channel_selector="average" # average two channels
            ).samples

            features.append(feature)

        audio_inputs = self.processor(features, sampling_rate=16000, return_tensors="pt")
        
        # we use labels for calculating loss
        # "target" is the dataset key name
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'audio_features': audio_inputs.get("input_features"),
            'audio_attention_mask': audio_inputs.get("attention_mask"),
            'audio_positions': torch.stack(audio_positions),

            'context_input_ids': context_inputs['input_ids'],
            'context_attention_mask': context_inputs['attention_mask'],

            # for debugging
            'contexts': [item['context'] for item in batch],
            'targets': [item['target'] for item in batch],
            
            "metadata": [item for item in batch],
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    