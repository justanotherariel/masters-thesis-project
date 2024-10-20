"""Main dataset"""
from typing import Any
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from src.typing.pipeline_objects import XData
from src.modules.training.models.transformer import SOS_TOKEN, SEP_TOKEN, PAD_TOKEN

class TorchDataset(Dataset):
    """Main dataset for transformer training with token-level processing."""
        
    def __init__(self, data: XData, indices: str) -> None:
        """Set up the dataset for training."""
        if indices != 'all_indices' and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")
        
        data.check_data()
        self.data = data
        self.indices = (getattr(data, indices) if indices != 'all_indices' 
                       else np.array(range(len(data.x_states))))
        
        # Calculate total number of possible token combinations per sample
        self.tokens_per_state = np.prod(data.x_states.shape[1:-1])  # x * y
        self.total_input_tokens = self.tokens_per_state + 1  # states + action
        self.total_output_tokens = self.tokens_per_state + 1  # states + reward
        self.combinations_per_sample = self.total_output_tokens  # Each output token is a target

    def __len__(self) -> int:
        """Get the total number of token-level training examples."""
        return len(self.indices) * self.combinations_per_sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single token-level training example more efficiently.
        
        Returns:
            tuple: (input_sequence, target_token)
        """
        if self.data is None or not self.data.check_data():
            raise ValueError("Dataset not initialized.")
        
        # Pre-allocate input sequence array
        input_tokens = torch.empty((self.total_input_tokens + self.total_output_tokens, 3), dtype=torch.int8)
        
        # Calculate indices
        sample_idx = self.indices[idx // self.combinations_per_sample]
        position_idx = idx % self.combinations_per_sample
        
        # Fill constant tokens
        input_tokens[0] = SOS_TOKEN
        input_tokens[self.total_input_tokens - 2] = torch.tensor([100, 1, self.data.x_actions[sample_idx].item()], dtype=torch.int8)
        input_tokens[self.total_input_tokens - 1] = SEP_TOKEN
        
        # Efficiently copy state data
        x_state_view = self.data.x_states[sample_idx].reshape(-1, 3)
        y_state_view = self.data.y_states[sample_idx].reshape(-1, 3)
        
        input_tokens[1:self.tokens_per_state + 1] = torch.tensor(x_state_view)
        input_tokens[self.total_input_tokens:self.total_input_tokens + position_idx] = torch.tensor(
            y_state_view[:position_idx]
        )
                
        # TODO: Suport for variable length sequences
        # Create final input sequence view up to current position
        # input_sequence = input_tokens[:self.total_input_tokens + position_idx]
        input_tokens[self.total_input_tokens + position_idx:] = PAD_TOKEN
        input_sequence = input_tokens
        
        # Determine target
        if position_idx < self.tokens_per_state:
            target = torch.tensor(y_state_view[position_idx])
        else:
            target = torch.tensor([100, 1, self.data.y_rewards[sample_idx].item()], dtype=torch.int8)
            
        return input_sequence, target