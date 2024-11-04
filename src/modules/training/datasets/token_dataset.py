"""Main dataset"""

from dataclasses import dataclass

from typing import Any, Callable, Optional
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.environment.gymnasium import flatten_indices
from src.typing.pipeline_objects import XData

from enum import Enum

class TokenType(Enum):
    PAD = 0
    OBSERVATION = 1
    ACTION = 2
    REWARD = 3
    SOS = 4
    SEP = 5

from typing import Any, Dict, List, Tuple, Union
import torch

class TokenIndexSub:
    """Handles indexing for a specific token type."""
    def __init__(self, instance: 'TokenIndex', name: str):
        self.name = name
        self.instance = instance
        
    def _get_single(self, key: int) -> torch.Tensor:
        """Get tensor indices for a single key."""
        info = self.instance.info[self.name]
        info_discrete = self.instance.info_discrete[self.name]
        
        if self.instance.discrete:
            start_idx = info_discrete[key][0]
            end_idx = (start_idx + info_discrete[key][1]) if info_discrete[key][1] != 0 else (start_idx + 1)
            return torch.arange(start_idx, end_idx)
        return torch.tensor([info[key][0]])
    
    def __getitem__(self, key: Union[int, slice]) -> torch.Tensor:
        """Get tensor indices for the given key or slice."""
        info = self.instance.info[self.name]
        
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(info) if key.stop is None else key.stop
            
            indices = [self._get_single(i) for i in range(start, stop)]
            return torch.cat(indices) if indices else torch.tensor([])
            
        return self._get_single(key)
    
    def __len__(self) -> int:
        """Return the number of indices for this token type."""
        return len(self.instance.info[self.name])

class TokenIndex:
    """Manages token indices for different types defined in the info dictionary."""
    def __init__(self, info: Dict[str, List[Tuple[int, int]]]):
        required_keys = {'type', 'observation', 'action', 'reward'}
        if not required_keys.issubset(info.keys()):
            raise ValueError(f"TokenIndex must have all keys: {required_keys}")    
        
        # dict[type, list[tuple[original_idx, num_items]]]
        self.info: Dict[str, List[Tuple[int, int]]] = info
        
        # dict[type, list[tuple[discrete_start_idx, len]]]
        self.info_discrete: Dict[str, List[Tuple[int, int]]] = {}
        
        self.discrete: bool = False
        self._calculate_discrete_info()
    
    def _calc_num_items_before(self, org_idx: int) -> int:
        """Calculate total number of items before a given original index."""
        num_items = 0
        for i in range(org_idx):
            for value in self.info.values():
                orig_indices = [x[0] for x in value]
                if i in orig_indices:
                    idx = orig_indices.index(i)
                    num_items += value[idx][1]
        return num_items
    
    def _calculate_discrete_info(self) -> None:
        """Calculate discrete index information for all token types."""
        self.info_discrete = {
            key: [(self._calc_num_items_before(original_idx), num_items) 
                 for original_idx, num_items in value]
            for key, value in self.info.items()
        }
    
    def __getattr__(self, name: str) -> Any:
        """
        Enable dynamic access to TokenIndexSub instances and full token sequences.
        Supports both direct access (e.g., type) and underscore suffix (e.g., type_).
        """
        # Handle direct attribute access (e.g., type, observation)
        if name in self.info:
            return TokenIndexSub(self, name)
            
        # Handle underscore suffix access (e.g., type_, observation_)
        if name.endswith('_') and name[:-1] in self.info:
            return TokenIndexSub(self, name[:-1])[:]
            
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    @property
    def shape(self) -> int:
        """Return the total number of token indices."""
        if self.discrete:
            return sum([index[1] if index[1] != 0 else 1 for sub in self.info.values() for index in sub])
        return sum([len(x) for x in self.info.values()])
    
    def get_discrete_idx(self, org_idx: int) -> tuple[int, int]:
        """Get the discrete index and number of items for a given original index"""
        for key, value in self.info.items():
            for i, (idx, num_items) in enumerate(value):
                if idx == org_idx:
                    return self.info_discrete[key][i]
        raise ValueError(f"Original index {org_idx} out of range.")

class TokenMinigridDataset(Dataset):
    """Main dataset for transformer training with token-level processing."""

    discretize: bool = False

    _data: XData | None = None
    ti: TokenIndex | None = None
    _indices: npt.NDArray | None = None
    _data_len_of_state: int | None = None
    _data_len_of_input: int | None = None
    _data_len_of_output: int | None = None
    _token_combinations: int | None = None

    def __init__(self, data: XData, indices: str, discretize: bool = False) -> None:
        """Set up the dataset for training."""
        if indices != "all_indices" and not hasattr(data, indices):
            raise ValueError(f"Data does not have attribute {indices}")
        
        data.check_data()
        self._data = data
        self.discretize = discretize

        # Calculate total number of possible token combinations per sample
        self._data_len_of_state = np.prod(data.observations.shape[1:-1])  # x * y
        self._data_len_of_input = 1 + self._data_len_of_state + 1 + 1  # SOS  + observations + action + SEP
        self._data_len_of_output = self._data_len_of_state + 1  # observations + reward
        self._token_combinations = self._data_len_of_output  # Each output token is a target

        # Grab coorect indices
        self._indices = getattr(data, indices) if indices != "all_indices" else np.array(range(len(data.observations)))
        self._indices = flatten_indices(self._indices)

        # Expand indices to account for all possible token combinations
        self._token_indices = np.arange(len(self._indices) * self._token_combinations)
    
    @staticmethod
    def create_ti(info: dict[str, Any]) -> TokenIndex:
        """Create a TokenIndex object from the given info dictionary."""
        observation_info = info['env_build']['observation_info']
        action_info = info['env_build']['action_info']
        reward_info = info['env_build']['reward_info']

        token_info = {
            'type': [(0, len(TokenType))],
        }
        start_idx = 1

        token_info.update({
            'observation': [(start_idx + idx, num_items) for (idx, num_items) in observation_info],
        })
        start_idx += len(observation_info)
        
        token_info.update({
            'action': [(start_idx + idx, num_items) for (idx, num_items) in action_info],
        })
        start_idx += 1
        
        token_info.update({
            'reward': [(start_idx + idx, num_items) for (idx, num_items) in reward_info],
        })
        
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
        # Each token has 6 dimensions: special token value, observation 1, observation 2, observation 3, action, reward
        self.ti.discrete = False
        x = torch.zeros((self._data_len_of_input + self._data_len_of_output - 1, 6), dtype=torch.uint8)
        x[0, self.ti.type_] = TokenType.SOS.value  # Start of sequence
        x[1 : self._data_len_of_state + 1, self.ti.type_] = TokenType.OBSERVATION.value
        x[1 : self._data_len_of_state + 1, self.ti.observation_] = torch.tensor(
            self._data.observations[sample_idx[0]].reshape(-1, 3)
        )  # Observation
        x[self._data_len_of_input - 2, self.ti.type_] = TokenType.ACTION.value  # Action
        x[self._data_len_of_input - 2, self.ti.action_] = self._data.actions[sample_idx[2]].item() # Action
        x[self._data_len_of_input - 1, self.ti.type_] = TokenType.SEP.value  # End of sequence

        # Add the rest of the output tokens
        x[self._data_len_of_input : self._data_len_of_input + position_idx, self.ti.type_] = TokenType.OBSERVATION.value
        x[self._data_len_of_input : self._data_len_of_input + position_idx, self.ti.observation_] = torch.tensor(
            self._data.observations[sample_idx[1]].reshape(-1, 3)[:position_idx]
        )

        # Pad the rest of input token sequence (Already padded with zeros)
        # x[self._data_len_of_input + position_idx :, self.ti.type_] = TokenType.PAD.value

        # Determine target
        y = torch.zeros((6, ), dtype=torch.uint8)
        if position_idx < self._data_len_of_state:
            y[self.ti.type_] = TokenType.OBSERVATION.value
            y[self.ti.observation_] = torch.tensor(self._data.observations[sample_idx[1]].reshape(-1, 3)[position_idx], dtype=torch.uint8)
        else:
            y[self.ti.type_] = TokenType.REWARD.value
            y[self.ti.reward_] = self._data.rewards[sample_idx[2]].item()
            
        if self.discretize:
            x, y = self.discretizer(x, y)

        return x, y

class TokenDiscretizer:
    """Discretize the tokens."""
    
    def __init__(self, ti: TokenIndex):
        self.ti = ti
        
        self.ti.discrete = True
        self.new_shape = ti.shape
        
    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.ti.discrete = True
        new_x = torch.zeros((x.shape[0], self.new_shape), dtype=x.dtype)
        new_y = torch.zeros((self.new_shape, ), dtype=y.dtype)
        new_idx = 0
        
        for org_idx in range(x.shape[1]):
            new_idx, num_classes = self.ti.get_discrete_idx(org_idx)
            
            if num_classes == 0:
                new_x[:, new_idx] = x[:, org_idx]
                new_y[new_idx] = y[org_idx]
                continue
            
            new_x[:, new_idx:new_idx+num_classes] = torch.nn.functional.one_hot(x[:, org_idx].long(), num_classes=num_classes)
            new_y[new_idx:new_idx+num_classes] = torch.nn.functional.one_hot(y[org_idx].long(), num_classes=num_classes)
                
        return new_x, new_y
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.apply(*args, **kwds)