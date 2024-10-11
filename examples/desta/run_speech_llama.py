
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
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, TypeVar, Union, cast

# overwrite detector for interactive SLURM in our cluster
# SLURMEnvironment.detect = lambda: False

# overwrite pl strategy to load model state dict with strict=False
def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
    assert self.lightning_module is not None
    self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

pl.strategies.Strategy.load_model_state_dict = load_model_state_dict


@hydra_runner(config_path="conf", config_name="whisper_llama")
def main(cfg):
    pl.seed_everything(42)

    trainer = pl.Trainer(
        callbacks=[RichModelSummary(max_depth=4)],
        **cfg.trainer
    )

    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f"\n\nConfig: {OmegaConf.to_yaml(cfg)}\n\n")
    model = SpeechLLaMA(cfg, trainer)

    OmegaConf.save(model.cfg, f"{log_dir}/config.yaml")

    trainer.fit(model)

    
if __name__ == '__main__':
    main()