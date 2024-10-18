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
    test_indices: NDArray[np.int32] | None