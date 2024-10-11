
class MultiSpeechLlamaDataset(SpeechLlamaDataset):
    """
    Support multiple audio inputs.
    """

    def __init__(self, cfg, data_cfg):
        super().__init__(cfg, data_cfg)

    def batchified_preprocess_function(self, examples):
        # Add data root to audio filepaths
        examples["audio_filepath"] = [str(self.data_cfg.data_root) + "/" + filepath for filepath in examples["audio_filepath"]]
        new_audio_filepath2 = []
        for example in examples["audio_filepath2"]:
            if example != None:
                new_audio_filepath2.append(str(self.data_cfg.data_root) + "/" + example)
            else:
                new_audio_filepath2.append(None)
        

        # Apply chat template to create contexts
        context = self.tokenizer.apply_chat_template(
            self.data_cfg.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format contexts with example data
        # - input
        # - transcription
        # - transcription2
        formatted_contexts = []
        for i in range(len(examples["audio_filepath"])):
            example_dict = {key: examples[key][i] for key in self.data_cfg.replace_keys}
            formatted_contexts.append(context.format(**example_dict))

        # Process each context to find audio position and split text
        audio_positions = []
        audio_positions2 = []
        
        new_contexts = []
        for context in formatted_contexts:
            assert len(context.split(self.cfg.dataset.audio_locator)) == 2, f"Audio locator {self.cfg.dataset.audio_locator} not found in inputs: {context}"
            left_text, right_text = context.split(self.cfg.dataset.audio_locator)
            audio_positions.append(len(self.tokenizer.tokenize(left_text)))

            context = left_text + right_text

            assert len(context.split("|audio_features2|")) == 2, f"Audio locator |audio_features2| not found in inputs: {context}"
            left_text, right_text = context.split("|audio_features2|")
            audio_positions2.append(len(self.tokenizer.tokenize(left_text)))
            context = left_text + right_text
            new_contexts.append(context)

        # Update examples with new data
        examples["audio_position"] = audio_positions
        examples["audio_positions2"] = audio_positions2
        examples["context"] = [context for context in new_contexts]
        examples["target"] = [target + self.tokenizer.eos_token for target in examples["target"]]

        return examples

    
    def collate_fn(self, batch):
        text_inputs = self.tokenizer([item['context']+item["target"] for item in batch], truncation=True, padding="longest", max_length=200, return_tensors="pt", return_length=True, add_special_tokens=False)
        context_inputs = self.tokenizer(
            [item['context'] for item in batch], truncation=True, padding="longest", max_length=200, return_tensors="pt", return_length=True, add_special_tokens=False
        )
        
        labels = torch.full_like(text_inputs['input_ids'], -100)
        features = []
        features2 = []
        has_audio2 = [] # list of boolean specify if the entry has audio2
        audio_positions = []
        audio_positions2 = []
        for i, item in enumerate(batch):
            # build labels (padding_side = left)
            total_length = text_inputs['length'][i]
            context_length = len(self.tokenizer.tokenize(item["context"]))
            
            pad_length = total_length - text_inputs["attention_mask"][i].sum()

            start_answer_position = pad_length + context_length
            labels[i, start_answer_position:] = text_inputs['input_ids'][i, start_answer_position:]

            # audio position
            audio_positions.append(item["audio_position"] + pad_length) # padding left
            audio_positions2.append(item["audio_positions2"] + pad_length) # padding left

            # load audios
            feature = AudioSegment.from_file(
                item["audio_filepath"],
                target_sr=16000,
                duration=item["duration"],
            ).samples
            features.append(feature)

            if item["audio_filepath2"] != None:
                feature2 = AudioSegment.from_file(
                    item["audio_filepath2"],
                    target_sr=16000,
                    duration=item["duration"],
                ).samples
                features2.append(feature2)
                has_audio2.append(1)
            else:
                has_audio2.append(0)


        features = self.processor(features, sampling_rate=16000, return_tensors="pt").input_features
        features2 = self.processor(features2, sampling_rate=16000, return_tensors="pt").input_features
        
        features = torch.cat([features, features2], dim=0)
        
        # we use labels for calculating loss
        # "target" is the dataset key name
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'audio_features': features,
            'has_audio2': [item for item in has_audio2],

            'audio_positions': torch.stack(audio_positions),
            'audio_positions2': torch.stack(audio_positions2),

            'context_input_ids': context_inputs['input_ids'],
            'context_attention_mask': context_inputs['attention_mask'],

            # for debugging
            'contexts': [item['context'] for item in batch],
            'targets': [item['target'] for item in batch],
            
            "metadata": [item for item in batch],
        }