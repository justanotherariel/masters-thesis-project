from dataclasses import dataclass, field
from enum import Flag
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from numpy.typing import NDArray

from src.modules.training.datasets.tensor_index import TensorIndex


class DatasetGroup(Flag):
    """The types of datasets."""

    NONE = 0
    TRAIN = 1
    VALIDATION = 2
    TEST = 4
    ALL = TRAIN | VALIDATION | TEST


@dataclass
class PipelineData:
    env: gym.Env | None = None

    # Raw sampled Data
    observations: NDArray[np.number] | None = None
    actions: NDArray[np.int8] | None = None
    rewards: NDArray[np.float32] | None = None

    # Metadata
    indices: dict[DatasetGroup, NDArray[np.int32]] = field(default_factory=lambda: {})
    grids: dict[DatasetGroup, list[Grid]] = field(default_factory=lambda: {})

    # Predictions: list[observation, reward, other (optional - e.g. eta)]
    predictions: dict[DatasetGroup, list[Any]] = field(default_factory=lambda: {})
    model_last_epoch_recorded: int = 0
    model_training_time_s: float = 0.0
    accuracies: dict[DatasetGroup, dict[str, float]] = field(default_factory=lambda: {})

    def check_data(self) -> None:
        """Check if the data is valid. Can raise Errors."""

        # Check if all data is None or present
        if (
            self.observations is None
            or self.actions is None
            or self.rewards is None
            or self.indices is None
            or self.grids is None
        ):
            return

        if self.observations is None:
            raise ValueError("x_states is None.")
        if self.actions is None:
            raise ValueError("x_actions is None.")
        if self.rewards is None:
            raise ValueError("y_rewards is None.")
        if self.indices is None:
            raise ValueError("indices is None.")
        if self.grids is None:
            raise ValueError("grids is None.")

        # Check that keys of indices and grids match
        if set(self.indices.keys()) != set(self.grids.keys()):
            raise ValueError("indices keys do not match grids keys.")

        # Check if all lengths are the same
        length = len(self.actions)
        if len(self.rewards) != length:
            raise ValueError("rewards[] length does not match rewards[] length.")

        # Check keys of predictions if not None
        if self.predictions is not None and not set(self.predictions.keys()).issubset(self.indices.keys()):
            raise ValueError("predictions keys are not a subset of indices keys.")


@dataclass
class PipelineInfo:
    debug: bool = False
    output_dir: Path | None = None

    # Metadata about the observation, action, and reward data provided by the sampler
    data_info: dict[str, Any] | None = None

    # Model/Architecture specific information (TorchTrainer)
    trial_idx: int = -1  # The trial index during Random Seeds Initialization
    model_ti: TensorIndex | None = None  # TensorIndex object for the model
    model_train_on_discrete: bool = False  # Whether the model should train on discrete data
