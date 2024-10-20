import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


class XData:
    env: gym.Env | None
    x_states: NDArray[np.number] | None
    x_actions: NDArray[np.int8] | None
    y_states: NDArray[np.number] | None
    y_rewards: NDArray[np.float32] | None

    train_indices: NDArray[np.int32] | None
    validation_indices: NDArray[np.int32] | None

    validation_predictions: NDArray[np.number] | None

    def check_data(self) -> bool:
        """Check if the data is valid. Can raise Errors."""

        # Check if all data is not None
        if self.x_states is None:
            raise ValueError("x_states is None.")
        if self.x_actions is None:
            raise ValueError("x_actions is None.")
        if self.y_states is None:
            raise ValueError("y_states is None.")
        if self.y_rewards is None:
            raise ValueError("y_rewards is None.")

        # Check if all lengths are the same
        length = len(self.x_states)
        if len(self.x_actions) != length:
            raise ValueError("x_actions length does not match x_states length.")
        if len(self.y_states) != length:
            raise ValueError("y_states length does not match x_states length.")
        if len(self.y_rewards) != length:
            raise ValueError("y_rewards length does not match x_states length.")

        return True
