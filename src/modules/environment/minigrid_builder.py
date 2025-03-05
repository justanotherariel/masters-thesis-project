"""Block to instatiate a Gymnasium Minigrid Environment."""

from dataclasses import dataclass

import gymnasium as gym
import minigrid
import minigrid.core
import minigrid.core.grid
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.wrappers import NoDeath

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import PipelineData, PipelineInfo

from .minigrid_wrappers import FullyObsWrapper

logger = Logger()


def flatten_indices(indices: list[npt.NDArray[np.int32]]) -> npt.NDArray[np.int32]:
    """Flatten the indices for the dataset (remove the trajectory idx dimension) and remove padding."""
    indices = [ind.reshape(-1, 3) for ind in indices]
    return np.concatenate(indices, axis=0)


@dataclass
class MinigridBuilder(TransformationBlock):
    """Block to instatiate a Gymnasium Environment.

    :param environment: The environment to instantiate. E.g. MiniGrid-Empty-5x5-v0
    """

    environment: str

    def __post_init__(self):
        """Initialize the Gymnasium Environment."""
        self.env = gym.make(self.environment)

        # Check if the environment is a minigrid environment
        if not issubclass(type(self.env.unwrapped), minigrid.minigrid_env.MiniGridEnv):
            raise ValueError("Currently only MiniGrid environments are supported.")

        # Make Lava give negative rewards instead of termintating the episode
        self.env = NoDeath(self.env, no_death_types=("lava",), death_cost=-1.0)

        self.env = FullyObsWrapper(self.env)  # Output fully observable grid as numpy array
        self.env.reset(seed=42)  # Seed the environment

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """

        info.data_info = {
            "observation_space": self.env.observation_space,
            "observation_info": [
                (0, len(OBJECT_TO_IDX) - 1),  # Remove agent idx, represented by the last dimension
                (1, len(COLORS)),
                (2, len(STATE_TO_IDX)),
                (3, 5),  # 0: agent not present, 1-4: agent direction
            ],
            # "action_space": self.env.action_space,
            # "action_info": [ (0, self.env.action_space.n.item()) ],
            # Only Left/Right/Forward actions
            "action_space": spaces.Discrete(3),
            "action_info": [(0, 3)],
            "reward_info": [
                (0, 0),
            ],
        }

        return info

    def custom_transform(self, data: PipelineData) -> npt.NDArray[np.float32]:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        logger.info("Setting Environment")

        if data is None:
            data = PipelineData()

        data.env = self.env
        return data
