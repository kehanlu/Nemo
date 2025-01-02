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
import argparse
import torch

from nemo.utils.khlu import check_finename, check_consecutive_words

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Speech LLaMA model")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to evaluate")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--manifest_filepaths", type=str, required=True, help="Path to manifest file(s)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--config_file", type=str, default="", help="Path to config file")
    parser.add_argument("--overwrite_ckpt", type=str, default=None, help="Path to specific checkpoint file")

    args = parser.parse_args()
    
    # Convert "None" string to None
    if args.overwrite_ckpt == "None":
        args.overwrite_ckpt = None
    if args.config_file == "None":
        args.config_file = None
    return args

def main(args):
    pl.seed_everything(42)

    # ========================
    # Find checkpoint
    # ========================

    for ckpt in Path(args.exp_dir).glob("**/*.ckpt"):
        if f"epoch={args.epoch}" in str(ckpt):
            break
    else:
        assert False, f"Checkpoint not found for epoch={args.epoch} in {args.exp_dir}"

    logging.info(f"Loading checkpoint: {ckpt}")

    # ========================
    # Load model & predict
    # ========================
    model = SpeechLLaMA.load_from_checkpoint(ckpt, strict=False)

    if args.config_file and args.config_file != "None":
        model.cfg = OmegaConf.create(OmegaConf.load(args.config_file))
    
    model.cfg.trainer.devices = 1 # use one GPU
    trainer = pl.Trainer(**model.cfg.trainer)

    with open_dict(model.cfg):
        model.cfg.dataset.test_ds = model.cfg.dataset.validation_ds
        model.cfg.dataset.test_ds.manifest_filepaths = args.manifest_filepaths
        model.cfg.dataset.test_ds.data_root = args.data_root
        model.cfg.dataset.test_ds.batch_size = 8
    
    if args.overwrite_ckpt:
        logging.info("="*100)
        logging.info(f"Overwrite language model with {args.overwrite_ckpt}")
        logging.info(
            model.language_model.model.layers.load_state_dict(torch.load(args.overwrite_ckpt))
        )
        logging.info("="*100)
    logging.info(f"\n\nConfig: {OmegaConf.to_yaml(model.cfg)}\n\n")

    # run prediction
    # Prepare dataloader
    dataloader = model._build_dataloader(model.cfg.dataset.test_ds)

    # Run prediction
    results = trainer.predict(model, dataloaders=dataloader)

    # Calculate performance
    outputs = model._calculate_performace(results=results, data_cfg=model.cfg.dataset.test_ds, ckpt=ckpt)
    
    # Prepare output directory
    os.makedirs(f"{model.cfg.save_dir}/results/{args.dataset_name}", exist_ok=True)
    output_path = check_finename(f"{model.cfg.save_dir}/results/{args.dataset_name}/epoch={args.epoch}.jsonl")
    
    # Write outputs to file
    model._write_outputs_to_file(outputs, output_path)
    
    # Log information
    logging.info(f"Write predictions to:\n\n {output_path}\n")
    logging.info(f"Exp: {args.exp_dir}")
    logging.info(f"Dataset name: {args.dataset_name}")


if __name__ == '__main__':
    args = parse_args()
    main(args)