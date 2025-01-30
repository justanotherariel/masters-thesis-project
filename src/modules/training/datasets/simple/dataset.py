from dataclasses import dataclass
from enum import Enum, auto

import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from ..tensor_index import TensorIndex

@dataclass
class SimpleDataset(Dataset):
    """
    Simple Dataset that preserves 2D structure of observations and separates actions/rewards.
    """

    _data: PipelineData
    _indices: npt.NDArray

    def __init__(self, data: PipelineData, ds_group: DatasetGroup, discretize: bool = False) -> None:
        """Set up the dataset for training."""
        if ds_group not in data.indices:
            raise ValueError(f"Data does not have attribute {ds_group.name}.")

        self.discretize = discretize
        self._data = data
        self._data.check_data()

        # Fetch shape
        self._obs_shape = data.observations.shape[1:]  # (height, width, channels)

        # Grab correct indices
        self._indices = data.indices[ds_group]
        self._indices = flatten_indices(self._indices)

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
        # Grab the correct index. idx = [obs_x_idx, obs_y_idx, action/reward_idx]
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
            action = self._discretize_value(action, 'action')
            reward = self._discretize_value(reward, 'reward')

        return (x_obs, action), (y_obs, reward)

    def _discretize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Discretize a 2D observation while preserving its spatial structure."""
        if not hasattr(self, "ti"):
            raise ValueError("TokenIndex not initialized. Call setup() first.")

        original_shape = obs.shape
        flat_obs = obs.reshape(-1, obs.shape[-1])   # Reshape to (H*W, C)

        discretized = torch.zeros((flat_obs.shape[0], len(self.base_ti.observation_)), dtype=torch.uint8)
        for i in range(flat_obs.shape[1]):
            idx = self.base_ti.observation[i]
            if self.base_ti.info["observation"][i][1] > 0:  # If discretization is needed
                discretized[:, idx] = torch.nn.functional.one_hot(
                    flat_obs[:, i].long(), num_classes=self.base_ti.info["observation"][i][1]
                ).to(torch.uint8)
            else:
                discretized[:, idx] = flat_obs[:, i]

        return discretized.reshape(*original_shape[:-1], -1)    # Reshape back to (H, W, C')

    def _discretize_value(self, value: torch.Tensor, field_type: str) -> torch.Tensor:
        """Discretize a single value (action or reward)."""
        if not hasattr(self, "ti"):
            raise ValueError("TokenIndex not initialized. Call setup() first.")

        num_classes = self.base_ti.info[field_type][0][1]

        if num_classes > 0:
            return torch.nn.functional.one_hot(value.long(), num_classes=num_classes)[0].to(torch.uint8)
        return value

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param info: The configuration information.
        :return: The configuration information.
        """
        self.base_ti = self.create_base_ti(info)
        self.base_ti.discrete = self.discretize
        
        self.ti = self.create_ti(info)
        self.ti.discrete = self.discretize

        return info
    
    @staticmethod
    def create_base_ti(info: PipelineInfo) -> TensorIndex:
        """Create a TensorIndex object from the given info dictionary."""
        observation_info = info.data_info["observation_info"]
        action_info = info.data_info["action_info"]
        reward_info = info.data_info["reward_info"]

        token_info = {}

        token_info.update({"observation": observation_info})
        token_info.update({"action": action_info})
        token_info.update({"reward": reward_info})

        return TensorIndex(token_info)
