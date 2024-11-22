import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from typing import Any


class XData:
    env: gym.Env | None
    observations: NDArray[np.number] | None
    actions: NDArray[np.int8] | None
    rewards: NDArray[np.float32] | None

    train_indices: NDArray[np.int32] | None
    validation_indices: NDArray[np.int32] | None

    train_predictions: list[Any] | None
    train_targets: list[Any] | None
    validation_predictions: list[Any] | None
    validation_targets: list[Any] | None

    def check_data(self) -> bool:
        """Check if the data is valid. Can raise Errors."""

        # Check if all data is not None
        if self.observations is None:
            raise ValueError("x_states is None.")
        if self.actions is None:
            raise ValueError("x_actions is None.")
        if self.rewards is None:
            raise ValueError("y_rewards is None.")

        # Check if all lengths are the same
        length = len(self.actions)
        if len(self.rewards) != length:
            raise ValueError("rewards[] length does not match rewards[] length.")
        return True
