"""Block to instatiate a Gymnasium Minigrid Environment."""

from dataclasses import dataclass

import gymnasium as gym
import minigrid
import numpy as np
import numpy.typing as npt
import tqdm
from minigrid.wrappers import ImgObsWrapper

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import XData

from .minigrid_wrappers import FullyObsWrapper

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

        self.env = FullyObsWrapper(self.env)  # Output fully observable grid
        self.env = ImgObsWrapper(self.env)  # Output only numpy array
        self.env.reset(seed=42)  # Seed the environment

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
        train_samples = self.num_samples * self.perc_train

        # Sample the Environment
        observations: list[npt.NDArray] = []
        data.actions = np.empty((self.num_samples, 1), dtype=np.int8)
        data.rewards = np.empty((self.num_samples, 1), dtype=np.float16)
        
        # [trajectory_id, [state_x, state_y, action/reward_idx]]
        train_indices: list[list[int]] = []
        validation_indices: list[list[int]] = []
        
        samples_idx = 0
        state_idx = 0

        progress_bar = tqdm.tqdm(total=self.num_samples, desc="Sampling Environment", unit="samples")
        while samples_idx < self.num_samples:
            trajectory: list[int] = []
            observation, _info = env.reset()
            
            for i in range(self.num_samples_per_env):
                # Save current State
                observations.append(observation)

                # Take a step
                action = env.action_space.sample()
                observation, reward, terminated, truncated, _info = env.step(action)

                # Save Action and Reward
                data.actions[samples_idx] = action
                data.rewards[samples_idx] = reward
                
                # Save the indices
                trajectory.append((len(observations) - 1, len(observations), samples_idx))

                # Misc
                progress_bar.update(1)
                samples_idx += 1
                if samples_idx >= train_samples or samples_idx >= self.num_samples:
                    break
                if terminated or truncated:
                    break
                
            # Record the final state of the trajectory
            observations.append(observation)

            # Append the indices to the correct list
            indices_collected = np.full((self.num_samples_per_env, 3), -1)
            indices_collected[:len(trajectory)] = np.array(trajectory)
            if samples_idx <= train_samples:
                train_indices.append(indices_collected)
            else:
                validation_indices.append(indices_collected)

        data.observations = np.stack(observations, axis=0)
        data.train_indices = np.stack(train_indices, axis=0)
        data.validation_indices = np.stack(validation_indices, axis=0)

        env.close()
        return data
