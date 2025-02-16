"""Set seed for reproducibility."""

import torch

from src.framework.logging import Logger
import random
import numpy as np

logger = Logger()


def set_torch_seed(seed: int = 42, deterministic=True) -> None:
    """Set torch seed for reproducibility.

    :param seed: seed to set

    :return: None
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA's random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make CUDA operations deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_deterministic_debug_mode(1 if deterministic else 0)
    # torch.use_deterministic_algorithms(deterministic)