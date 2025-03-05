"""Main dataset"""

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.minigrid_builder import flatten_indices
from src.typing.pipeline_objects import PipelineData, PipelineInfo

from .tensor_index import TensorIndex, TokenDiscretizer, TokenType


class AutoregressiveTokenDataset(Dataset):
    """Main dataset for transformer training with token-level processing."""

    discretize: bool = False

    _data: PipelineData | None = None
    ti: TensorIndex | None = None
    _indices: npt.NDArray | None = None
    _data_len_of_obs: int | None = None
    _data_len_of_input: int | None = None
    _data_len_of_output: int | None = None
    _token_combinations: int | None = None

    def __init__(self, data: PipelineData, indices: str, discretize: bool = False) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")

        data.check_data()
        self._data = data
        self.discretize = discretize

        # Calculate total number of possible token combinations per sample
        self._data_len_of_obs = np.prod(data.observations.shape[1:-1])  # x * y
        self._data_len_of_input = 1 + self._data_len_of_obs + 1 + 1  # SOS  + observations + action + SEP
        self._data_len_of_output = self._data_len_of_obs + 1  # observations + reward
        self._token_combinations = self._data_len_of_output  # Each output token is a target

        # Grab coorect indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

        # Expand indices to account for all possible token combinations
        self._token_indices = np.arange(len(self._indices) * self._token_combinations)

    @staticmethod
    def create_ti(info: PipelineInfo) -> TensorIndex:
        """Create a TokenIndex object from the given info dictionary."""
        observation_info = info.data_info["observation_info"]
        action_info = info.data_info["action_info"]
        reward_info = info.data_info["reward_info"]

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

        return TensorIndex(token_info)

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """

        self.ti = self.create_ti(info)

        if self.discretize:
            self.discretizer = TokenDiscretizer(self.ti)

        return info

    def __len__(self) -> int:
        """Get the total number training examples."""
        return len(self._token_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example.

        Returns:
            tuple: (input_sequence, target_token)
        """
        if self._data is None:
            raise ValueError("Dataset not initialized.")
        self._data.check_data()

        # Calculate indices
        sample_idx = self._indices[idx // self._token_combinations]
        position_idx = idx % self._token_combinations

        # Create input token sequence
        self.ti.discrete = False
        x = torch.zeros((self._data_len_of_input + self._data_len_of_output - 1, self.ti.shape), dtype=torch.uint8)

        # SOS
        x[0, self.ti.type_] = TokenType.SOS.value

        # Add Initial Observation
        x[1 : self._data_len_of_obs + 1, self.ti.type_] = TokenType.OBSERVATION.value
        x[1 : self._data_len_of_obs + 1, self.ti.observation_] = torch.tensor(
            self._data.observations[sample_idx[0]].reshape(-1, self.ti.observation_.shape[0])
        )

        # Add Action
        x[self._data_len_of_input - 2, self.ti.type_] = TokenType.ACTION.value
        x[self._data_len_of_input - 2, self.ti.action_] = self._data.actions[sample_idx[2]].item()

        # Add SEP
        x[self._data_len_of_input - 1, self.ti.type_] = TokenType.SEP.value

        # Add resulting observation
        x[self._data_len_of_input : self._data_len_of_input + position_idx, self.ti.type_] = TokenType.OBSERVATION.value
        x[self._data_len_of_input : self._data_len_of_input + position_idx, self.ti.observation_] = torch.tensor(
            self._data.observations[sample_idx[1]].reshape(-1, self.ti.observation_.shape[0])[:position_idx]
        )

        # Pad the rest of input token sequence (Already padded with zeros)
        # x[self._data_len_of_input + position_idx :, self.ti.type_] = TokenType.PAD.value

        # Determine target
        y = torch.zeros((self.ti.shape,), dtype=torch.uint8)
        if position_idx < self._data_len_of_obs:
            y[self.ti.type_] = TokenType.OBSERVATION.value
            y[self.ti.observation_] = torch.tensor(
                self._data.observations[sample_idx[1]].reshape(-1, self.ti.observation_.shape[0])[position_idx]
            )
        else:
            y[self.ti.type_] = TokenType.REWARD.value
            y[self.ti.reward_] = self._data.rewards[sample_idx[2]].item()

        if self.discretize:
            x, y = self.discretizer(x, y)

        return x, y
