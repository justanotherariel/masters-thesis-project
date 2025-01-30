from typing import Any, Union

import torch

class TensorIndexSub:
    """Handles indexing for a specific type."""

    def __init__(self, instance: "TensorIndex", name: str):
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


class TensorIndex:
    """Manages indices as defined in the info dictionary."""

    def __init__(self, info: dict[str, list[tuple[int, int]]]):
        required_keys = {"observation", "action", "reward"}
        if not required_keys.issubset(info.keys()):
            raise ValueError(f"TokenIndex must have all keys: {required_keys}")

        # dict[type, list[tuple[original_idx, num_items]]]
        self.info: dict[str, list[tuple[int, int]]] = info

        # dict[type, list[tuple[discrete_start_idx, len]]]
        self.info_discrete: dict[str, list[tuple[int, int]]] = {}

        # Flags
        self.discrete: bool = False
        self.seperate: bool = False

        # Check if seperate
        num_idx_zero = 0
        for _key, value in self.info.items():
            indices = [x[0] for x in value]
            if 0 in indices:
                num_idx_zero += 1
        if num_idx_zero == len(self.info):
            self.seperate = True
        elif num_idx_zero != 1:
            raise ValueError("Only one or all keys can have index 0.")

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

    def _calc_num_items_before_seperate(self, key: str, org_idx: int) -> int:
        """Calculate total number of items before a given original index."""
        num_items = 0
        for i in range(org_idx):
            orig_indices = [x[0] for x in self.info[key]]
            if i in orig_indices:
                idx = orig_indices.index(i)
                num_items += self.info[key][idx][1]
        return num_items

    def _calculate_discrete_info(self) -> None:
        """Calculate discrete index information."""
        if not self.seperate:
            self.info_discrete = {
                key: [(self._calc_num_items_before(original_idx), num_items) for original_idx, num_items in value]
                for key, value in self.info.items()
            }
        else:
            self.info_discrete = {
                key: [
                    (self._calc_num_items_before_seperate(key, original_idx), num_items)
                    for original_idx, num_items in value
                ]
                for key, value in self.info.items()
            }

    def __getattr__(self, name: str) -> Any:
        """
        Enable dynamic access to TensorIndexSub instances.
        Supports both direct access (e.g., type) and underscore suffix (e.g., type_).
        """
        if name == "info":
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        # Handle direct attribute access (e.g., type, observation)
        if name in self.info:
            return TensorIndexSub(self, name)

        # Handle underscore suffix access (e.g., type_, observation_)
        if name.endswith("_") and name[:-1] in self.info:
            return TensorIndexSub(self, name[:-1])[:]

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
