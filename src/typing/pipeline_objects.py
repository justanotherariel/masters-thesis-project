import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class XData:
    env: gym.Env | None
    observations: NDArray[np.number] | None
    actions: NDArray[np.int8] | None
    rewards: NDArray[np.float32] | None

    train_indices: NDArray[np.int32] | None
    validation_indices: NDArray[np.int32] | None

    validation_predictions: NDArray[np.number] | None

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
        length = len(self.observations)
        if len(self.actions) != length:
            raise ValueError("x_actions length does not match x_states length.")
        if len(self.rewards) != length:
            raise ValueError("y_rewards length does not match x_states length.")

        return True
