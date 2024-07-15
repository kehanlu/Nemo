
import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.collections.desta.models.speech_llama import SpeechLLaMA

from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.plugins import environments
import typer
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import RichModelSummary

SLURMEnvironment.detect = lambda: False


@hydra_runner(config_path="conf", config_name="whisper_llama")
def main(cfg):
    pl.seed_everything(42)

    trainer = pl.Trainer(callbacks=[
        RichModelSummary(max_depth=3),
    ],**cfg.trainer)

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f"\n\nConfig: {OmegaConf.to_yaml(cfg)}\n\n")
    model = SpeechLLaMA(cfg, trainer)

    OmegaConf.save(model.cfg, f"{log_dir}/config.yaml")

    trainer.fit(model)

    


if __name__ == '__main__':
    main()