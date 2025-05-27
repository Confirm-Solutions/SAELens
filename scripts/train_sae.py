import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

load_dotenv()


def resolve_device(device_config: str) -> str:
    """Resolve device configuration."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Resolve environment variables in the config."""
    # Handle wandb entity and project from environment
    if cfg.wandb_entity is None:
        cfg.wandb_entity = os.getenv("WANDB_ENTITY")
    if cfg.wandb_project is None:
        cfg.wandb_project = os.getenv("WANDB_PROJECT")

    # Resolve device
    cfg.device = resolve_device(cfg.device)

    return cfg


def hydra_cfg_to_sae_cfg(cfg: DictConfig) -> LanguageModelSAERunnerConfig:
    """Convert Hydra config to SAE config."""
    # Resolve environment variables and device
    cfg = resolve_env_vars(cfg)

    # Convert OmegaConf to dict and then to SAE config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Ensure cfg_dict is a dictionary
    assert isinstance(cfg_dict, dict), "Configuration must be a dictionary"

    # Remove Hydra-specific keys and template variables that aren't part of SAE config
    keys_to_remove = ["defaults", "model_size"]
    for key in keys_to_remove:
        cfg_dict.pop(key, None)

    return LanguageModelSAERunnerConfig(**cfg_dict)  # type: ignore


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert to SAE config
    sae_cfg = hydra_cfg_to_sae_cfg(cfg)

    runner = SAETrainingRunner(sae_cfg)
    runner.run()


if __name__ == "__main__":
    main()
