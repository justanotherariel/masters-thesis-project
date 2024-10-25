"""Main dataset"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.modules.training.models.transformer import PAD_TOKEN, SEP_TOKEN, SOS_TOKEN
from src.typing.pipeline_objects import XData


@dataclass
class TokenMinigridDataset(Dataset):
    """Main dataset for transformer training with token-level processing."""

    _data: XData | None = None
    _indices: npt.NDArray | None = None
    _data_len_of_state: int | None = None
    _data_len_of_input: int | None = None
    _data_len_of_output: int | None = None
    _token_combinations: int | None = None

    def __init__(self, data: XData, indices: str) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")

        data.check_data()
        self._data = data

        # Calculate total number of possible token combinations per sample
        self._data_len_of_state = np.prod(data.observations.shape[1:-1])  # x * y
        self._data_len_of_input = 1 + self._data_len_of_state + 1 + 1  # SOS  + states + action + SEP
        self._data_len_of_output = self._data_len_of_state + 1  # states + reward
        self._token_combinations = self._data_len_of_output  # Each output token is a target

        # Grab coorect indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

        # Expand indices to account for all possible token combinations
        self._token_indices = np.arange(len(self._indices) * self._token_combinations)

    def __len__(self) -> int:
        """Get the total number training examples."""
        return len(self._token_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example.

        Returns:
            tuple: (input_sequence, target_token)
        """
        if self._data is None or not self._data.check_data():
            raise ValueError("Dataset not initialized.")

        # Calculate indices
        sample_idx = self._indices[idx // self._token_combinations]
        position_idx = idx % self._token_combinations

        # Create input token sequence
        x = torch.empty((self._data_len_of_input + self._data_len_of_output - 1, 3), dtype=torch.uint8)
        x[0] = SOS_TOKEN  # Start of sequence
        x[1 : self._data_len_of_state + 1] = torch.tensor(
            self._data.observations[sample_idx[0]].reshape(-1, 3)
        )  # State
        x[self._data_len_of_input - 2] = torch.tensor([100, 1, self._data.actions[sample_idx[2]].item()])  # Action
        x[self._data_len_of_input - 1] = SEP_TOKEN  # End of sequence

        # Add the rest of the output tokens
        x[self._data_len_of_input : self._data_len_of_input + position_idx] = torch.tensor(
            self._data.observations[sample_idx[1]].reshape(-1, 3)[:position_idx]
        )

        # TODO: Suport for variable length sequences by rewriting collate_fn
        # Pad the rest of input token sequence
        x[self._data_len_of_input + position_idx :] = PAD_TOKEN

        # Determine target
        if position_idx < self._data_len_of_state:
            y = torch.tensor(self._data.observations[sample_idx[1]].reshape(-1, 3)[position_idx])
        else:
            y = torch.tensor([100, 2, self._data.rewards[sample_idx[2]].item()])

        return x, y
