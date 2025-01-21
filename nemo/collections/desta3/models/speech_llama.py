from nemo.collections.common.models import ModelPT, Exportable
from transformers import AutoConfig, AutoModelForCausalLM, MllamaConfig, MllamaForCausalLM
from nemo.utils import logging
from nemo.collections.desta3.data.audio_dataset import AudioTextDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import torch
import gc
import json

class SpeechLlama(ModelPT, Exportable):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.cfg = cfg

        # ========================
        # add HF model config to cfg
        # ========================
        if self.cfg.model.language_model.model_id == "kehanlu/llm32":
            self.cfg.model.language_model.cfg = MllamaConfig.from_pretrained(cfg.model.language_model.model_id).to_dict()
        else:
            self.cfg.model.language_model.cfg = AutoConfig.from_pretrained(cfg.model.language_model.model_id).to_dict()
        self.cfg.model.speech_encoder.cfg = AutoConfig.from_pretrained(cfg.model.speech_encoder.model_id).to_dict()


        # ========================
        # Initialize langauge model
        # - Causal LM
        # - Lora
        # - Tokenizer
        # ========================
        if self.cfg.model.language_model.model_id == "kehanlu/llm32":
            self.language_model = MllamaForCausalLM.from_pretrained(
                self.cfg.model.language_model.model_id, torch_dtype=torch.bfloat16, cache_dir=os.getenv("HF_HOME"))
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.language_model.model_id, torch_dtype=torch.bfloat16, cache_dir=os.getenv("HF_HOME"))
        

        if hasattr(self.cfg.model, "lora") and self.cfg.model.lora is not None:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=self.cfg.model.lora.rank,
                target_modules=["q_proj", 'k_proj', "v_proj"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.language_model = get_peft_model(
                self.language_model,
                lora_config
            ).base_model.model

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # ========================
        # Initialize speech perception module
        # - Encoder + Modality connector
        # ========================
        if "whisper" in self.cfg.model.speech_encoder.model_id:
            self.perception = WhisperPerceptionModule(cfg=self.cfg)
        elif "w2v-bert-2.0" in self.cfg.model.speech_encoder.model_id:
            self.perception = SpeechPerceptionModule(cfg=self.cfg)
        else:
            raise NotImplementedError(f"model_id {self.cfg.model.speech_encoder.model_id} not implemented")


        # ========================
        # Setup optimizer
        # configure_optimizers() calls:
        # - setup_optimizer()
        # - setup_optimizer_param_groups()
        # ========================
        
        self.configure_optimizers()
        logging.info(f"********************** Model summary **********************\n")
        logging.info(f"\n{self.language_model}")
        logging.info(f"\n{self.summarize(max_depth=4)}")


        # ========================
        # Helpers
        # store intermediate outputs from training, validation, and prediction steps
        # ========================
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.prediction_step_outputs = []

    def forward(self, batch):
        inputs = self.prepare_llm_input(batch)
        outputs = self.language_model(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        return outputs

    def prepare_llm_input(self, batch):
        """
        return: 
            inputs_embeds, attention_mask, labels
        """
        input_ids = batch["text"]["input_ids"]
        attention_mask = batch["text"]["attention_mask"]
        labels = batch["text"]["labels"]

        bs = input_ids.size(0)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]

        audio_features, audio_feature_lengths = self.perception(
            input_features=batch["audio"]["input_features"],
            attention_mask=batch["audio"]["attention_mask"]
        )

        new_input_ids = []
        new_inputs_embeds = []
        new_attention_mask = []
        new_labels = []
        
        audio_position_shift = torch.zeros([bs], dtype=torch.long, device=self.device)

        # Inject audio features into the text_embeddings
        for feature_idx, (batch_idx, position) in enumerate(batch["text"]["audio_positions"]):
            audio_p = position + audio_position_shift[batch_idx]
            audio_feature = audio_features[feature_idx]
            audio_feature_length = audio_feature_lengths[feature_idx]

            audio_position_shift[batch_idx] += audio_feature_length
            
            item_input_ids = torch.cat([input_ids[batch_idx, :audio_p], torch.ones([audio_feature_length], dtype=torch.long, device=self.device)*128008, input_ids[batch_idx, audio_p:]], dim=0)
            item_inputs_embeds = torch.cat([inputs_embeds[batch_idx, :audio_p], audio_feature, inputs_embeds[batch_idx, audio_p:]], dim=0)
            item_attention_mask = torch.cat([attention_mask[batch_idx, :audio_p], torch.ones([audio_feature_length], dtype=torch.long, device=self.device), attention_mask[batch_idx, audio_p:]], dim=0)
            item_labels = torch.cat([labels[batch_idx, :audio_p], torch.full([audio_feature_length], -100, dtype=torch.long, device=self.device), labels[batch_idx, audio_p:]], dim=0)

            new_input_ids.append(item_input_ids)
            new_inputs_embeds.append(item_inputs_embeds)
            new_attention_mask.append(item_attention_mask)
            new_labels.append(item_labels)

        new_inputs_embeds = self._left_padding(new_inputs_embeds)
        new_attention_mask = self._left_padding(new_attention_mask)
        new_labels = self._left_padding(new_labels, -100)

        return {
            "inputs_embeds": new_inputs_embeds,
            "attention_mask": new_attention_mask,
            "labels": new_labels
        }
    
    def _left_padding(self, list_of_tensors, padding_value=0):
        batch_size = len(list_of_tensors)
        max_length = max([tensor.size(0) for tensor in list_of_tensors])

        if len(list_of_tensors[0].size()) == 0:
            padded = torch.full((batch_size, max_length), padding_value)
            for i, seq in enumerate(list_of_tensors):
                seq_len = seq.size(0)
                padded[i, -seq_len:] = seq
        else:
            dim = list_of_tensors[0].size(1)
            padded = torch.full((batch_size, max_length, dim), padding_value)
            # Fill in sequences from the right
            for i, seq in enumerate(list_of_tensors):
                seq_len = seq.size(0)
                padded[i, -seq_len:] = seq
            
        return padded

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        batch_size = batch["text"]["input_ids"].size(0)
        self.log("train_loss", loss, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self.log("train_ppl", perplexity, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=batch_size)
        self.log(
            'global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=batch_size
        )

        self.training_step_outputs.append({'train_loss': loss.item(), 'train_ppl': perplexity.item()})

        # for monitoring
        if batch_idx % self.cfg.model.debug.train_log_every_n_steps == 0:
            self.predict_step(batch, batch_idx)
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        batch_size = batch["text"]["input_ids"].size(0)
        self.log("val_loss", loss, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self.log("val_ppl", perplexity, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)

        self.validation_step_outputs.append({"val_loss": loss, "val_ppl": perplexity})
        return {"val_loss": loss, "val_ppl": perplexity}

    def predict_step(self, batch, batch_idx):
        self.eval()

        inputs = self.prepare_llm_input(batch)
        outputs = self.language_model.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=self.cfg.model.generation_config.do_sample,
            max_new_tokens=self.cfg.model.generation_config.max_new_tokens,
            temperature=self.cfg.model.generation_config.temperature,
            top_p=self.cfg.model.generation_config.top_p,
        )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = []
        for context, pred, target, metadata in zip(batch["text"]["contexts"], predictions, batch["text"]["targets"], batch["metadata"]):
            result = {
                "context": context,
                "prediction": pred,
                "label": self.tokenizer.decode(
                    self.tokenizer.encode(target, add_special_tokens=False), skip_special_tokens=True
                ) # remove special tokens
            }
            result.update(metadata)
            results.append(result)


        # ========================
        # Write intermediate predictions to file for debugging
        # ========================
        with open(f"{self.cfg.save_dir}/predictions.jsonl", "a") as fo:
            fo.write(json.dumps(result)+ "\n")
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        return results


    def on_train_epoch_end(self):
        logging.info("********************** Training epoch end **********************")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):

        # write predictions
        dataset_name = "val"
        os.makedirs(f"{self.cfg.save_dir}/results/{dataset_name}", exist_ok=True)
        output_path = f"{self.cfg.save_dir}/results/{dataset_name}/val@{self.trainer.global_step}-{self.trainer.current_epoch}.jsonl"

        results = [batch["preds"] for batch in self.validation_step_outputs]
        outputs = self._calculate_performace(results=results, data_cfg=self.cfg.dataset.validation_ds, ckpt=f"ep={self.trainer.current_epoch}-{self.trainer.global_step}")
        self._write_outputs_to_file(outputs, output_path)

        self.validation_step_outputs.clear()

    def train_dataloader(self):
        data_cfg = self.cfg.dataset.train_ds
        logging.info("\n********************* Training dataset *********************\n")
        dataloader = self._build_dataloader(data_cfg)
        logging.info("\n***************** End of Training dataset *****************\n")
        
        self._train_dl = dataloader # for ModelPT
        return dataloader

    def val_dataloader(self):
        data_cfg = self.cfg.dataset.validation_ds
        logging.info("\n******************** Validation dataset ********************\n")
        dataloader = self._build_dataloader(data_cfg)
        logging.info("\n**************** End of Validation dataset ****************\n")
        
        self._validation_dl = dataloader # for ModelPT
        return dataloader
    
    def _build_dataloader(self, data_cfg):
        """
        helper function
        """
        dataset = AudioTextDataset(cfg=self.cfg, data_cfg=data_cfg)
        logging.info(dataset[0])
        dataloader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            collate_fn=dataset.collate_fn, 
            shuffle=data_cfg.shuffle,
            pin_memory=data_cfg.pin_memory,
        )
        return dataloader

    def setup_training_data(self, data):
        # Nemo
        pass

    def setup_validation_data(self, data):
        # Nemo
        pass

    def list_available_models(self):
        # Implementation of the method
        return ["model1", "model2", "model3"]
    

    # ==== Nemo ModelPT ====
    def configure_optimizers(self):
        # Nemo
        # overwrite ModelPT.configure_optimizers
        self.setup_optimization(self.cfg.model.optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]
        
    def setup_optimizer_param_groups(self):
        # Nemo
        # follow megatron style
        self.unfreeze()
        known_groups = []

        if self.cfg.model.language_model.freeze == True:
            for param in self.language_model.parameters():
                param.requires_grad = False
            known_groups.append('language_model.')
        
        if self.cfg.model.speech_encoder.freeze == True:
            for param in self.perception.encoder.parameters():
                param.requires_grad = False
            known_groups.append('perception.encoder.')

        opt_params = []
        opt_params_name = []
        for n, p in self.named_parameters():
            is_unknown = True
            for group in known_groups:
                if n.startswith(group):
                    is_unknown = False
            if is_unknown:
                opt_params_name.append(n)
                opt_params.append(p)
                logging.info(f"\nTrainable: {n} {p.size()}")

        for n, p in self.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
                opt_params_name.append(n)
                opt_params.append(p)
                logging.info(f"\nTrainable: {n} {p.size()}")

        self._optimizer_param_groups = [
            {"params": opt_params}
        ]

        self._optimizer_param_groups_name = opt_params_name # @khlu: for saving trainable state_dict only


    def state_dict(self):
        # overwrite torch.nn.Module.state_dict
        # only save updated parameters

        state_dict = []
        for n, p in self.named_parameters():
            if n in self._optimizer_param_groups_name:
                state_dict.append((n, p))

        state_dict = OrderedDict(state_dict)
        return state_dict
    
    def _load_pretrained_weights(self):
        """
        Load pretrained weights from the cfg.
        Ignore the missing keys.
        """
        logging.info(f"********************** Load pre-trained weights **********************\nFrom{self.cfg.model.restore_from_path}\n")
        logging.info(self.load_state_dict(
            torch.load(self.cfg.model.restore_from_path)["state_dict"],
            strict=False
        ))
        logging.info(f"********************** End of load pre-trained weights **********************\n")