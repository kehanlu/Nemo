from nemo.collections.desta.models.speech_llama import SpeechLLaMA
from pytorch_lightning.plugins.environments import SLURMEnvironment
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.utils import logging
import typer
SLURMEnvironment.detect = lambda: False
import json
from collections import defaultdict
from whisper_normalizer.basic import BasicTextNormalizer
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from nemo.utils.khlu import check_finename, check_consecutive_words

def main(exp_dir:str = typer.Option(..., "--exp_dir"), 
         epoch:int = typer.Option(..., "--epoch"),
         dataset_name:str = typer.Option(..., "--dataset_name"),
         manifest_filepaths:str = typer.Option(..., "--manifest_filepaths"),
         data_root:str = typer.Option(..., "--data_root"),
         config_file:str = typer.Option("", "--config_file")
        ):
    pl.seed_everything(42)

    # ========================
    # Find checkpoint
    # ========================
    for ckpt in Path(exp_dir).glob("**/*.ckpt"):
        if f"epoch={epoch}" in str(ckpt):
            break
    else:
        assert False, f"Checkpoint not found for epoch={epoch} in {exp_dir}"
    logging.info(f"Loading checkpoint: {ckpt}")

    # ========================
    # Load model & predict
    # ========================
    model = SpeechLLaMA.load_from_checkpoint(ckpt, strict=False)

    if config_file and config_file != "None":
        model.cfg = OmegaConf.create(OmegaConf.load(config_file))

    logging.info(f"\n\nConfig: {OmegaConf.to_yaml(model.cfg)}\n\n")
    model.cfg.trainer.devices = 1 # use one GPU

    # OmegaConf.save(model.cfg, f"{model.cfg.save_dir}/config.yaml")

    trainer = pl.Trainer(**model.cfg.trainer)

    with open_dict(model.cfg):
        model.cfg.dataset.test_ds = model.cfg.dataset.validation_ds
        model.cfg.dataset.test_ds.manifest_filepaths = manifest_filepaths
        model.cfg.dataset.test_ds.data_root = data_root
        model.cfg.dataset.test_ds.batch_size = 8


    # run prediction
    dataloader = model._build_dataloader(model.cfg.dataset.test_ds)
    results = trainer.predict(model, dataloaders=dataloader)

    outputs = model._calculate_performace(results=results, data_cfg=model.cfg.dataset.test_ds, ckpt=ckpt)
    
    os.makedirs(f"{model.cfg.save_dir}/results/{dataset_name}", exist_ok=True)
    output_path = check_finename(f"{model.cfg.save_dir}/results/{dataset_name}/epoch={epoch}.jsonl")
    model._write_outputs_to_file(outputs, output_path)
    
    
    logging.info(f"write predictions to:\n\n {output_path}\n")
    logging.info(f"Exp: {exp_dir}")
    logging.info(f"Dataset name: {dataset_name}")



if __name__ == '__main__':
    typer.run(main)