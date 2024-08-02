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

from collections import defaultdict
from whisper_normalizer.basic import BasicTextNormalizer
from nemo.utils.khlu import check_finename, check_consecutive_words
from omegaconf import DictConfig, OmegaConf, open_dict

from peft import LoraConfig, TaskType, get_peft_model
from collections import OrderedDict

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
        
        # ========================
        # add HF model config for language model and speech encoder
        # ========================
        self.cfg.model.language_model.cfg = AutoConfig.from_pretrained(cfg.model.language_model.model_id).to_dict()
        self.cfg.model.speech_encoder.cfg = AutoConfig.from_pretrained(cfg.model.speech_encoder.model_id).to_dict()


        # ========================
        # Initialize langauge model
        # - Causal LM
        # - Lora
        # ========================
        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.language_model.model_id, torch_dtype=torch.bfloat16,
            cache_dir="/NeMo/.cache"
        )
        
        
        if hasattr(self.cfg.model, "lora") and self.cfg.model.lora is not None:
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
        self.perception = WhisperPerceptionModule(cfg=self.cfg)


        # ========================
        # Load pre-trained weights if "restore_from_path" is provided
        # ========================
        if self.cfg.model.restore_from_path is not None:
            self._load_pretrained_weights()


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
        
        # for monitoring
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

        # this is original LLM, without audio inputs.
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

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        results = []
        for context, pred, target, metadata in zip(batch["contexts"], predictions, batch["targets"], batch["metadata"]):
            result = {
                "context": context,
                "prediction": pred,
                "target": self.tokenizer.decode(
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
        dataset = SpeechLlamaDataset(cfg=self.cfg, data_cfg=data_cfg)
        logging.info(dataset[0])
        dataloader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            collate_fn=dataset.collate_fn, 
            shuffle=data_cfg.shuffle,
            pin_memory=data_cfg.pin_memory,
        )
        return dataloader
    
    # ==== Nemo ModelPT ====
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
        # state_dict = self.perception.connector.state_dict(
        #     prefix="perception.connector."
        # )
        
        # params = []
        # for n, p in self.named_parameters():
        #     if "lora_" in n:
        #         params.append((n, p))
        # state_dict.update(params)
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


    # ========================
    # Related to performance calculation and writing outputs to file
    # ========================
    def _dict_of_lists_to_list_of_dicts(self, d): 
        """
        Convert a dictionary of lists to a list of dictionaries
        """
        return [{k: v[i] for k, v in d.items()} for i in range(len(next(iter(d.values()))))]

    
    def _calculate_performace(self, results, data_cfg, ckpt=None):
        """
        Calculate performance based on the question types and metrics.
        
        Parameters:
        -----------
        results: List of List of Dict
            - dict:
                - question_type
                - metric
                - prediction
                - target

        Returns:
        --------
        ouputs: Dict
            Contains info and prediction results. Ready to dump into a json file.
        """
        

        normalizer = BasicTextNormalizer()
        question_groups = defaultdict(lambda: {"correct": 0, "total": 0})
        outputs = []
        for i, batch in enumerate(results):
            # batch: [{}, {}]
            for result in batch:
                question_type = result["question_type"]
                metric = result["metric"]

                prediction = normalizer(result["prediction"].replace("<|eot_id|>", ""))
                target = normalizer(result["target"].replace("<|eot_id|>", ""))

                if metric == "accuracy":
                    if check_consecutive_words(text=prediction, words=target):
                        question_groups[question_type]["correct"] += 1
                        result["correct"] = True
                    else:
                        result["correct"] = False
                question_groups[question_type]["total"] += 1
                result["index"] = len(outputs)
                outputs.append(result)
        accuracies = {}
        for question_type, counts in question_groups.items():
            accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            accuracies[question_type] = accuracy

        easy_copy_format = ",".join(
            [f"{v:.4f}" for k, v in accuracies.items()]
        )
        return {
            "info": {
                "accuracy": accuracies,
                "total_data": len(outputs),
                "dataset": OmegaConf.to_container(data_cfg, resolve=True),
                "ckpt": str(ckpt),
                "easy_copy_format": easy_copy_format
            },
            "results": outputs,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
    

    def _write_outputs_to_file(self, outputs: dict, output_path: str):
        """
        Parameters:
        -----------
        outputs: Dict
        output_path: str
            Path to the output file. Make sure the parent directory exists.

        Returns:
        --------
        result: {
            info: {}
            results: [{}] # list of predictions
        }
        """

        output_path = check_finename(output_path)
        
        with open(output_path, "w") as fo:
            json.dump(outputs, fo, indent=2, ensure_ascii=False)
        
        logging.info(f"write predictions to:\n\n{output_path}\n")
        logging.info("********************** End of write file **********************")