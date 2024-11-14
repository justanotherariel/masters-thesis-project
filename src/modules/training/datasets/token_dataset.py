"""Main dataset"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.typing.pipeline_objects import XData
from .utils import TokenDiscretizer, TokenIndex, TokenType


@dataclass
class TokenDataset(Dataset):
    """Simple dataset for transformer training."""

    _data: XData | None = None
    _indices: npt.NDArray | None = None

    def __init__(self, data: XData, indices: str, discretize: bool = False) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")
        
        self.discretize = discretize

        data.check_data()
        self._data = data

        # Calculate input/output shapes
        self._data_len_of_obs = np.prod(data.observations.shape[1:-1])  # x * y
        self._data_len_of_input = self._data_len_of_obs + 1  # states + action
        self._data_len_of_output = self._data_len_of_obs + 1  # states + reward

        # Grab coorect indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

    def __len__(self) -> int:
        """Get the total number training examples."""
        return self._indices.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example.

        Returns:
            tuple: (input_sequence, target_token)
        """
        if self._data is None or not self._data.check_data():
            raise ValueError("Dataset not initialized.")

        # Grab the correct index / [state_x, state_y, action/reward_idx]
        idx = self._indices[idx]

        # Create input token sequence
        self.ti.discrete = False
        x = torch.zeros((self._data_len_of_input, self.ti.shape), dtype=torch.uint8)
        
        # Add Initial Observation
        x[: self._data_len_of_obs, self.ti.type_] = TokenType.OBSERVATION.value
        x[: self._data_len_of_obs, self.ti.observation_] = torch.tensor(
            self._data.observations[idx[0]].reshape(-1, self.ti.observation_.shape[0])
        )
        
        # Add Action
        x[-1, self.ti.type_] = TokenType.ACTION.value
        x[-1, self.ti.action_] = self._data.actions[idx[2]].item()


        # Create target token sequence
        y = torch.zeros((self._data_len_of_input, self.ti.shape), dtype=torch.uint8)
        
        # Add Resulting Observation
        y[: self._data_len_of_obs, self.ti.type_] = TokenType.OBSERVATION.value
        y[: self._data_len_of_obs, self.ti.observation_] = torch.tensor(
            self._data.observations[idx[1]].reshape(-1, self.ti.observation_.shape[0])
        )
        
        # Add Reward
        y[-1, self.ti.type_] = TokenType.REWARD.value
        y[-1, self.ti.reward_] = self._data.rewards[idx[2]].item()

        if self.discretize:
            x, y = self.discretizer(x, y)

        return x, y

    @staticmethod
    def create_ti(info: dict[str, Any]) -> TokenIndex:
        """Create a TokenIndex object from the given info dictionary."""
        observation_info = info["env_build"]["observation_info"]
        action_info = info["env_build"]["action_info"]
        reward_info = info["env_build"]["reward_info"]

        token_info = {
            "type": [(0, len(TokenType))],
        }
        start_idx = 1

        token_info.update(
            {
                "observation": [(start_idx + idx, num_items) for (idx, num_items) in observation_info],
            }
        )
        start_idx += len(observation_info)

        token_info.update(
            {
                "action": [(start_idx + idx, num_items) for (idx, num_items) in action_info],
            }
        )
        start_idx += len(action_info)

        token_info.update(
            {
                "reward": [(start_idx + idx, num_items) for (idx, num_items) in reward_info],
            }
        )

        return TokenIndex(token_info)

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """

        self.ti = self.create_ti(info)

        if self.discretize:
            self.discretizer = TokenDiscretizer(self.ti)

        return info
