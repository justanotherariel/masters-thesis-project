"""Schema for the train configuration."""

from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class TrainConfig:
    """Schema for the train configuration.

    :param model: The model pipeline.
    :param ensemble: The ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param test_size: Size of the test set.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    """

    wandb: WandBConfig

    model: Any

    # Cache Path - for sampled environemnts
    cache_path: str
    
    # Debug Parameter disables wandb and multi-threading
    debug: bool = False
    
    # Additional data for sweeps
    sweep_data: Any = None
    
    # Monte Carlo Initialization
    seed: int = 42
    trial_idx: int = -1             # Set during execution
    n_trials: int | None = None
