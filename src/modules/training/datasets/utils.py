from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch


class TokenType(Enum):
    PAD = 0
    OBSERVATION = 1
    ACTION = 2
    REWARD = 3
    SOS = 4
    SEP = 5


class TokenIndexSub:
    """Handles indexing for a specific token type."""

    def __init__(self, instance: "TokenIndex", name: str):
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
        required_keys = {"type", "observation", "action", "reward"}
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
            key: [(self._calc_num_items_before(original_idx), num_items) for original_idx, num_items in value]
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
        if name.endswith("_") and name[:-1] in self.info:
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
            for i, (idx, _num_items) in enumerate(value):
                if idx == org_idx:
                    return self.info_discrete[key][i]
        raise ValueError(f"Original index {org_idx} out of range.")

class TokenDiscretizer:
    """Discretize the tokens."""

    def __init__(self, ti: TokenIndex):
        self.ti = ti

        self.ti.discrete = True
        self.new_shape = ti.shape

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.ti.discrete = True
        new_x = torch.zeros((x.shape[0], self.new_shape), dtype=x.dtype)
        new_y = torch.zeros((self.new_shape,), dtype=y.dtype)
        new_idx = 0

        for org_idx in range(x.shape[1]):
            new_idx, num_classes = self.ti.get_discrete_idx(org_idx)

            if num_classes == 0:
                new_x[:, new_idx] = x[:, org_idx]
                new_y[new_idx] = y[org_idx]
                continue

            new_x[:, new_idx : new_idx + num_classes] = torch.nn.functional.one_hot(
                x[:, org_idx].long(), num_classes=num_classes
            )
            new_y[new_idx : new_idx + num_classes] = torch.nn.functional.one_hot(
                y[org_idx].long(), num_classes=num_classes
            )

        return new_x, new_y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.apply(*args, **kwds)
