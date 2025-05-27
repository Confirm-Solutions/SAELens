# ruff: noqa: E402
__version__ = "5.9.1"

import logging
import os

logger = logging.getLogger(__name__)


# Configure logging only if not already configured by external systems (like Hydra)
def _setup_logging():
    """Setup logging for SAE Lens if not already configured by external systems."""
    _log_level = os.getenv("SAE_LENS_LOG_LEVEL", "INFO").upper()
    if hasattr(logging, _log_level):
        logger.setLevel(getattr(logging, _log_level))

    # Check if logging is already configured (e.g., by Hydra)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # External logging system is active, don't add our own handlers
        return

    # Only add handler if no external logging system is detected
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s:%(lineno)d → %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


_setup_logging()

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    PretokenizeRunnerConfig,
)
from .evals import run_evals
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .sae import SAE, SAEConfig
from .sae_training_runner import SAETrainingRunner
from .toolkit.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .training.activations_store import ActivationsStore
from .training.training_sae import TrainingSAE, TrainingSAEConfig
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "SAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
    "PretrainedSaeHuggingfaceLoader",
    "PretrainedSaeDiskLoader",
]
