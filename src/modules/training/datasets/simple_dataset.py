"""Main dataset"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.typing.pipeline_objects import XData


@dataclass
class SimpleMinigridDataset(Dataset):
    """Simple dataset for transformer training."""

    _data: XData | None = None
    _indices: npt.NDArray | None = None

    def __init__(self, data: XData, indices: str) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")

        data.check_data()
        self._data = data

        # Calculate input/output shapes
        self._data_len_of_state = np.prod(data.observations.shape[1:-1])  # x * y
        self._data_len_of_input = self._data_len_of_state + 1  # states + action
        self._data_len_of_output = self._data_len_of_state + 1  # states + reward

        # Grab coorect indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

    def __len__(self) -> int:
        """Get the total number training examples."""
        return len(self._indices)

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
        x = torch.empty((self._data_len_of_input, 3), dtype=torch.uint8)
        x[: self._data_len_of_state] = torch.tensor(self._data.observations[idx[0]].reshape(-1, 3))
        x[-1] = torch.tensor(self._data.actions[idx[2]])

        # Create target token sequence
        y = torch.empty((self._data_len_of_output, 3), dtype=torch.uint8)
        y[: self._data_len_of_state] = torch.tensor(self._data.observations[idx[1]].reshape(-1, 3))
        y[-1] = torch.tensor(self._data.rewards[idx[2]])

        return x, y
