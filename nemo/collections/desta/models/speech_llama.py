import json

from nemo.core.classes.exportable import Exportable
from nemo.core.classes import ModelPT
# from nemo.collections.nlp.models.nlp_model import NLPModel
import torch
import os
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperModel, BertConfig, WhisperForConditionalGeneration

from nemo.utils import logging
# from nemo.collections.desta.data.speech_llama_dataset import SpeechLlamaDataset
from nemo.collections.desta.data.simple_dataset import SpeechLlamaDataset
from nemo.collections.desta.modules.speech_encoder import WhisperPerceptionModule

from transformers import AutoConfig

import torch.nn as nn

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(self.size), torch.randn(self.size)


class SpeechLLaMA(ModelPT, Exportable):
    def __init__(self, cfg, trainer=None):
        self.cfg = cfg
        super().__init__(cfg=cfg, trainer=trainer)

        # add HF model config for language model and speech encoder
        self.cfg.model.language_model.cfg = AutoConfig.from_pretrained(cfg.model.language_model.model_id).to_dict()
        self.cfg.model.speech_encoder.cfg = AutoConfig.from_pretrained(cfg.model.speech_encoder.model_id).to_dict()


        self.language_model = AutoModelForCausalLM.from_pretrained(self.cfg.model.language_model.model_id, torch_dtype=torch.bfloat16)
        self.language_model.model.layers = self.language_model.model.layers[:8]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.language_model.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # Encoder
        self.perception = WhisperPerceptionModule(cfg=self.cfg)


        # load from checkpoint
        if self.cfg.model.restore_from_path:
            logging.info(f"Restoring from checkpoint: {self.cfg.model.restore_from_path}")
            logging.info(
                self.load_state_dict(torch.load(self.cfg.model.restore_from_path)
                ,strict=False)
            )
        

        self.configure_optimizers()
            # - setup_optimizer()
            # - setup_optimizer_param_groups()
        logging.info(f"********************** Model summary **********************\n{self.summarize(max_depth=3)}")
        


        # helper for averaging loss
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, batch):
        
        inputs_embeds, attention_mask, labels = self.prepare_llm_input(input_ids=batch["input_ids"], 
                               attention_mask=batch["attention_mask"],
                               labels=batch["labels"],
                               audio_features=batch["audio_features"],
                               audio_positions=batch["audio_positions"]
        )
        

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


    def prepare_llm_input(self, input_ids, attention_mask, labels, audio_features, audio_positions):
        # put audio features into the input
        bs = input_ids.size(0)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]
        
        
        audio_features, audio_feature_lengths = self.perception(audio_features)

        new_input_ids = []
        new_inputs_embeds = []
        new_attention_mask = []
        new_labels = []

        for i in range(bs):
            audio_p = audio_positions[i]
            audio_feature_length = audio_feature_lengths[i]
            new_input_ids.append(
                torch.cat([input_ids[i, :audio_p], torch.ones([audio_feature_length], dtype=torch.long, device=self.device), input_ids[i, audio_p:]], dim=0)
            )

            new_inputs_embeds.append(
                torch.cat([inputs_embeds[i, :audio_p], audio_features[i, :], inputs_embeds[i, audio_p:]], dim=0)
            )
            new_attention_mask.append(
                torch.cat([attention_mask[i, :audio_p], torch.ones([audio_feature_length], dtype=torch.long, device=self.device), attention_mask[i, audio_p:]], dim=0)
            )
            new_labels.append(
                torch.cat([labels[i, :audio_p], torch.full([audio_feature_length], -100, dtype=torch.long, device=self.device), labels[i, audio_p:]], dim=0)
            )

            # for idx, (inp, att, lab) in enumerate(zip(new_input_ids[i], new_attention_mask[i], new_labels[i])):
            #     inp, att, lab = inp.item(), att.item(), lab.item()

            #     print(f"({idx})", inp, att, lab, self.tokenizer.decode(inp).strip(), sep="\t")
            # print("="*100)
            

        new_inputs_embeds = torch.stack(new_inputs_embeds)
        new_attention_mask = torch.stack(new_attention_mask)
        new_labels = torch.stack(new_labels)
        return new_inputs_embeds, new_attention_mask, new_labels
        
    

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self(batch)

        # logging
        loss = outputs.loss
        perplexity = torch.exp(loss)
        batch_size = batch["input_ids"].size(0)
        self.log("train_loss", loss, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self.log("train_ppl", perplexity, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=batch_size)
        self.log(
            'global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=batch_size
        )

        self.training_step_outputs.append({'train_loss': loss, 'train_ppl': perplexity})
        
        if batch_idx % self.cfg.model.debug.train_log_every_n_steps == 0:
            self.predict_step(batch, batch_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        batch_size = batch["input_ids"].size(0)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        self.log("val_ppl", perplexity, sync_dist=True, batch_size=batch_size)
        
    
        preds = self.predict_step(batch, batch_idx)

        self.validation_step_outputs.append({"val_loss": loss, "val_ppl": perplexity, "preds": preds})
        return {"val_loss": loss, "val_ppl": perplexity, "preds": preds}

    def predict_step(self, batch, batch_idx):
        # Generation
        self.eval()

        # "context_input_ids"
        inputs_embeds, attention_mask, labels = self.prepare_llm_input(input_ids=batch["context_input_ids"], 
                               attention_mask=batch["context_attention_mask"],
                               labels=batch["labels"],
                               audio_features=batch["audio_features"],
                               audio_positions=batch["audio_positions"]
        )

        # inputs_embeds = self.language_model.model.embed_tokens(batch["context_input_ids"])
        # attention_mask = batch["context_attention_mask"]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=self.cfg.model.generation_config.do_sample,
            max_new_tokens=self.cfg.model.generation_config.max_new_tokens,
            temperature=self.cfg.model.generation_config.temperature,
            top_p=self.cfg.model.generation_config.top_p,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        if batch_idx:
            with open(f"{self.cfg.save_dir}/predictions.jsonl", "a") as fo:
                fo.write(json.dumps({
                    "context": batch["context"][0],
                    "response": responses[0].replace("<|eot_id|>", ""),
                    "answer": batch["answer"][0].replace("<|eot_id|>", ""),
                    "epoch": self.trainer.current_epoch,
                })+ "\n")


        preds = []
        for context, response, answer, metadata in zip(batch["context"], responses, batch["answer"], batch["metadata"]):
            metadata.update({
                "context": context,
                "response": response,
                "answer": answer,
            })
            preds.append(metadata)
        
        return preds
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["train_loss"] for x in self.training_step_outputs]).mean()
        avg_perplexity = torch.stack([x["train_ppl"] for x in self.training_step_outputs]).mean()
        # self.log('avg_train_loss', avg_loss, sync_dist=True, batch_size=1)
        # self.log('avg_train_perplexity', avg_perplexity, sync_dist=True, batch_size=1)
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_perplexity = torch.stack([x["val_ppl"] for x in self.validation_step_outputs]).mean()
        # self.log('avg_val_loss', avg_loss, sync_dist=True, batch_size=1)
        # self.log('avg_val_perplexity', avg_perplexity, sync_dist=True, batch_size=1)

        
        
        self._save_checkpoint(filename=f"ckpt-@val_loss{avg_loss:.3f}@val_ppl{avg_perplexity:.3f}@step{self.trainer.global_step}-{self.trainer.current_epoch}.pt")

        # write predictions
        dataset_name = "val"
        os.makedirs(f"{self.cfg.save_dir}/results/{dataset_name}", exist_ok=True)
        with open(f"{self.cfg.save_dir}/results/{dataset_name}/val@{self.trainer.global_step}-{self.trainer.current_epoch}.jsonl", "a") as fo:
            for item in self.validation_step_outputs:
                for pred in item["preds"]:
                    fo.write(json.dumps(pred) + "\n")
                

        self.validation_step_outputs.clear()

    def _save_checkpoint(self, filename):
        os.makedirs(f"{self.cfg.save_dir}/checkpoints", exist_ok=True)
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/checkpoints/{filename}")
        

    def _setup_dataloader_from_config(self):
        pass

    def setup_training_data(self, data):
        pass

    def setup_validation_data(self, data):
        pass

    def list_available_models(self):
        # Implementation of the method
        return ["model1", "model2", "model3"]
    

    def train_dataloader(self):     
        data_cfg = self.cfg.dataset.train_ds
        logging.info("\n********************* Training dataset *********************\n")
        dataset = SpeechLlamaDataset(cfg=self.cfg, data_cfg=data_cfg)
        logging.info(dataset[0])
        dataloader = DataLoader(dataset, batch_size=data_cfg.batch_size, collate_fn=dataset.collate_fn, 
        shuffle=data_cfg.shuffle)
        logging.info("\n***************** End of Training dataset *****************\n")
        
        
        self._train_dl = dataloader # for ModelPT
        return dataloader
    
    def val_dataloader(self):
        data_cfg = self.cfg.dataset.validation_ds
        logging.info("\n******************** Validation dataset ********************\n")
        dataset = SpeechLlamaDataset(cfg=self.cfg, data_cfg=data_cfg)
        logging.info(dataset[0])
        dataloader = DataLoader(dataset, batch_size=data_cfg.batch_size, collate_fn=dataset.collate_fn, shuffle=data_cfg.shuffle)
        logging.info("\n**************** End of Validation dataset ****************\n")
        
        
        self._validation_dl = dataloader # for ModelPT
        return dataloader
    
    
    # ==== kehan ====
    def configure_optimizers(self):
        # overwrite ModelPT.configure_optimizers
        self.setup_optimization(self.cfg.model.optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]
        
    def setup_optimizer_param_groups(self):
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
        for n, p in self.named_parameters():
            is_unknown = True
            for group in known_groups:
                if n.startswith(group):
                    is_unknown = False
            if is_unknown:
                opt_params.append(p)
                logging.info(f"\nTrainable: {n} {p.size()}")


        self._optimizer_param_groups = [
            {"params": opt_params}
        ]


    def state_dict(self):
        # only save updated parameters
        state_dict = self.perception.connector.state_dict(
            prefix="perception.connector."
        )
        return state_dict
    

    def _dict_of_lists_to_list_of_dicts(self, d): 
        """
        Convert a dictionary of lists to a list of dictionaries
        """
        return [{k: v[i] for k, v in d.items()} for i in range(len(next(iter(d.values()))))]
