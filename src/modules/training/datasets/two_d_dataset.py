from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.typing.pipeline_objects import XData

from .utils import TokenIndex, TokenType


@dataclass
class TwoDDataset(Dataset):
    """Dataset that preserves 2D structure of observations and separates actions/rewards."""

    _data: XData
    _indices: npt.NDArray
    _oversampling_factor: int

    def __init__(self, data: XData, indices: str, discretize: bool = False, oversampling_factor: int = 1) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")

        self.discretize = discretize

        data.check_data()
        self._data = data

        # Calculate observation shape
        self._obs_shape = data.observations.shape[1:-1]  # (height, width)
        self._obs_channels = data.observations.shape[-1]  # number of channels

        # Grab correct indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

        # Oversample the indices
        self._indices = np.repeat(self._indices, oversampling_factor, axis=0)

    def __len__(self) -> int:
        """Get the total number of training examples."""
        return self._indices.shape[0]

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Get a single training example.

        Returns:
            tuple: ((input_observation, action), (target_observation, reward))
                - input_observation: 2D tensor of shape (height, width, channels)
                - action: single value tensor
                - target_observation: 2D tensor of shape (height, width, channels)
                - reward: single value tensor
        """
        if self._data is None or not self._data.check_data():
            raise ValueError("Dataset not initialized.")

        # Grab the correct index / [state_x, state_y, action/reward_idx]
        idx = self._indices[idx]

        # Create input observation and action
        x_obs = torch.tensor(self._data.observations[idx[0]], dtype=torch.uint8)
        action = torch.tensor([self._data.actions[idx[2]].item()], dtype=torch.uint8)

        # Create target observation and reward
        y_obs = torch.tensor(self._data.observations[idx[1]], dtype=torch.uint8)
        reward = torch.tensor([self._data.rewards[idx[2]].item()], dtype=torch.float32)

        if self.discretize:
            x_obs = self._discretize_observation(x_obs)
            y_obs = self._discretize_observation(y_obs)
            action = self._discretize_value(action, TokenType.ACTION)
            reward = self._discretize_value(reward, TokenType.REWARD)

        return (x_obs, action), (y_obs, reward)

    def _discretize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Discretize a 2D observation while preserving its spatial structure."""
        if not hasattr(self, "ti"):
            raise ValueError("TokenIndex not initialized. Call setup() first.")

        original_shape = obs.shape
        # Reshape to (H*W, C) for discretization
        flat_obs = obs.reshape(-1, obs.shape[-1])

        self.ti.discrete = True
        discretized = torch.zeros((flat_obs.shape[0], len(self.ti.observation_)), dtype=torch.uint8)
        for i in range(flat_obs.shape[1]):
            idx = self.ti.observation[i]
            if self.ti.info["observation"][i][1] > 0:  # If discretization is needed
                discretized[:, idx] = torch.nn.functional.one_hot(
                    flat_obs[:, i].long(), num_classes=self.ti.info["observation"][i][1]
                ).to(torch.uint8)
            else:
                discretized[:, idx] = flat_obs[:, i]

        # Reshape back to (H, W, C')
        return discretized.reshape(*original_shape[:-1], -1)

    def _discretize_value(self, value: torch.Tensor, token_type: TokenType) -> torch.Tensor:
        """Discretize a single value (action or reward)."""
        if not hasattr(self, "ti"):
            raise ValueError("TokenIndex not initialized. Call setup() first.")

        if token_type == TokenType.ACTION:
            num_classes = self.ti.info["action"][0][1]
        else:  # TokenType.REWARD
            num_classes = self.ti.info["reward"][0][1]

        if num_classes > 0:
            return torch.nn.functional.one_hot(value.long(), num_classes=num_classes)[0].to(torch.uint8)
        return value

    @staticmethod
    def create_ti(info: dict[str, Any]) -> TokenIndex:
        """Create a TokenIndex object from the given info dictionary."""
        observation_info = info["env_build"]["observation_info"]
        action_info = info["env_build"]["action_info"]
        reward_info = info["env_build"]["reward_info"]

        token_info = {}

        token_info.update({"observation": observation_info})
        token_info.update({"action": action_info})
        token_info.update({"reward": reward_info})

        return TokenIndex(token_info)

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The configuration information.
        :return: The configuration information.
        """
        info.update({"train": {"dataset": self.__class__.__name__}})

        self.ti = self.create_ti(info)

        return info

    @staticmethod
    def custom_collate_fn(
        batch: list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Custom collate function for TwoDDataset that properly batches 2D observations with actions/rewards.

        Args:
            batch: List of ((x_obs, action), (y_obs, reward)) tuples
                - x_obs: shape (height, width, channels)
                - action: shape (action_dim,) or (num_classes,) if discretized
                - y_obs: shape (height, width, channels)
                - reward: shape (1,) or (num_classes,) if discretized

        Returns:
            tuple: ((x_obs_batch, action_batch), (y_obs_batch, reward_batch))
                - x_obs_batch: shape (batch_size, height, width, channels)
                - action_batch: shape (batch_size, action_dim) or (batch_size, num_classes)
                - y_obs_batch: shape (batch_size, height, width, channels)
                - reward_batch: shape (batch_size, 1) or (batch_size, num_classes)
        """
        # Unzip the batch into separate lists
        x_obs_list = []
        action_list = []
        y_obs_list = []
        reward_list = []

        for (x_obs, action), (y_obs, reward) in batch:
            x_obs_list.append(x_obs)
            action_list.append(action)
            y_obs_list.append(y_obs)
            reward_list.append(reward)

        # Stack the observations
        x_obs_batch = torch.stack(x_obs_list)  # (B, H, W, C)
        y_obs_batch = torch.stack(y_obs_list)  # (B, H, W, C)

        # Stack actions and rewards
        action_batch = torch.stack(action_list)  # (B, A) or (B, num_classes)
        reward_batch = torch.stack(reward_list)  # (B, 1) or (B, num_classes)

        return (x_obs_batch, action_batch), (y_obs_batch, reward_batch)
