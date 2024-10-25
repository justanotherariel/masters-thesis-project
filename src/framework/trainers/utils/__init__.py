"""Module with utility functions for training."""

from .get_dependencies import _get_onnxrt, _get_openvino
from .recursive_repr import recursive_repr

__all__ = [
    "_get_onnxrt",
    "_get_openvino",
    "recursive_repr",
]
