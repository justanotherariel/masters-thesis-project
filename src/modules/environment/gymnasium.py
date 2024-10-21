"""Block to instatiate a Gymnasium Minigrid Environment."""

from dataclasses import dataclass

import gymnasium as gym
import minigrid
import numpy as np
import numpy.typing as npt
import tqdm
from enum import Enum
from minigrid.wrappers import ImgObsWrapper

from .minigrid_wrappers import FullyObsWrapper
from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import XData

logger = Logger()

@dataclass
class GymnasiumBuilder(TransformationBlock):
    """Block to instatiate a Gymnasium Environment.

    :param environment: The environment to instantiate. E.g. MiniGrid-Empty-5x5-v0
    """

    environment: str

    def __post_init__(self):
        """Initialize the Gymnasium Environment."""
        if "MiniGrid-BlockedUnlockPickup-v0" not in gym.envs.registry:
            minigrid.register_minigrid_envs()
        self.env = gym.make(self.environment)
        
        # Check if the environment is a minigrid environment
        if not issubclass(type(self.env.unwrapped), minigrid.minigrid_env.MiniGridEnv):
            raise ValueError("Currently only MiniGrid environments are supported.")
        
        self.env = FullyObsWrapper(self.env)    # Output fully observable grid
        self.env = ImgObsWrapper(self.env)      # Output only numpy array
        self.env.reset(seed=42)                 # Seed the environment

    def custom_transform(self, data: XData) -> npt.NDArray[np.float32]:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        logger.info("Setting Environment")

        if data is None:
            data = XData()

        data.env = self.env
        return data


@dataclass
class GymnasiumSampler(TransformationBlock):
    """Block to sample a Gymnasium Environment.

    :param environment: The environment to instantiate. E.g. MiniGrid-Empty-5x5-v0
    """

    num_samples: int
    num_samples_per_env: int
    perc_train: float

    def custom_transform(self, data: XData) -> npt.NDArray[np.float32]:
        """Sample the environment.

        :param data: XData object containing the environment
        """
        env = getattr(data, "env", None)

        # Check if environment is initialized
        if env is None:
            raise ValueError("Environment is not initialized.")

        # Check if the environment is a minigrid environment
        if not issubclass(type(env.unwrapped), minigrid.minigrid_env.MiniGridEnv):
            raise ValueError("Currently only MiniGrid environments are supported.")

        logger.info("Sampling Environment")

        # Identify the important parameters
        grid_size = (env.unwrapped.width, env.unwrapped.height)

        # Sample the Environment
        data.x_states = np.empty((self.num_samples, *grid_size, 3), dtype=np.int8)
        data.x_actions = np.empty((self.num_samples, 1), dtype=np.int8)
        data.y_states = np.empty((self.num_samples, *grid_size, 3), dtype=np.int8)
        data.y_rewards = np.empty((self.num_samples, 1), dtype=np.float32)
        samples_collected = 0

        progress_bar = tqdm.tqdm(total=self.num_samples, desc="Sampling Environment", unit="samples")
        while samples_collected < self.num_samples:
            observation, _info = env.reset()
            for _ in range(self.num_samples_per_env):
                # Random Action
                action = env.action_space.sample()

                # Save State and Action Chosen
                data.x_states[samples_collected] = observation
                data.x_actions[samples_collected] = action

                # Take a step
                observation, reward, terminated, truncated, _info = env.step(action)

                # Record the next state and reward
                data.y_states[samples_collected] = observation
                data.y_rewards[samples_collected] = reward
                samples_collected += 1

                # Misc
                progress_bar.update(1)
                if samples_collected >= self.num_samples:
                    break
                if terminated or truncated:
                    break

        env.close()

        # Split the data into training and testing data
        # TODO: Take into consideration 'reserved' states, which are states
        # that are not allowed to be in the training data. These
        # should only be in the testing data.
        data.train_indices = np.random.choice(self.num_samples, int(self.perc_train * self.num_samples), replace=False)
        data.validation_indices = np.setdiff1d(np.arange(self.num_samples), data.train_indices)

        return data
