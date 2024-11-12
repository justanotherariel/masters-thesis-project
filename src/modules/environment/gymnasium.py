"""Block to instatiate a Gymnasium Minigrid Environment."""

from bdb import set_trace
from dataclasses import dataclass
from typing import Any, Tuple

import gymnasium as gym
import minigrid
import minigrid.core
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.wrappers import ImgObsWrapper

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import XData

from .minigrid_wrappers import FullyObsWrapper

logger = Logger()


def flatten_indices(indices: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Flatten the indices for the dataset (remove the trajectory idx dimension) and remove padding."""
    indices_flat = indices.reshape(-1, 3)
    return indices_flat[indices_flat[:, 0] != -1]


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

        self.env = FullyObsWrapper(self.env)  # Output fully observable grid as numpy array
        self.env.reset(seed=42)  # Seed the environment

    def setup(self, data: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """
        if data is None:
            data = {}

        data.update(
            {
                "env_build": {
                    "action_space": self.env.action_space,
                    "observation_space": self.env.observation_space,
                    "observation_info": [
                        (0, len(OBJECT_TO_IDX) -1), # Remove agent idx, represented by the last dimension
                        (1, len(COLORS)),
                        (2, len(STATE_TO_IDX)),
                        (3, 5),  # 0: agent not present, 1-4: agent direction
                    ],
                    "action_info": [
                        (0, self.env.action_space.n.item()),
                    ],
                    "reward_info": [
                        (0, 0),
                    ],
                }
            }
        )

        return data

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
class GymnasiumSamplerRandom(TransformationBlock):
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

        progress_bar = tqdm(total=self.num_samples, desc="Sampling Environment", unit="samples")
        while samples_idx < self.num_samples:
            trajectory: list[int] = []
            observation, _info = env.reset()

            for _ in range(self.num_samples_per_env):
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
            indices_collected[: len(trajectory)] = np.array(trajectory)
            if samples_idx <= train_samples:
                train_indices.append(indices_collected)
            else:
                validation_indices.append(indices_collected)

        data.observations = np.stack(observations, axis=0)
        data.train_indices = np.stack(train_indices, axis=0)
        data.validation_indices = np.stack(validation_indices, axis=0)

        env.close()
        return data

@dataclass
class MinigridSamplerExtensive(TransformationBlock):
    """Block to extensively sample a MiniGrid Environment by placing the agent at each valid position and executing each possible action.
    Requires the environment to be a MiniGrid environment and that the last wrapper provides the function getObservation(), which returns the current observation without taking a step.

    Args:
        train_envs (int): Number of environments to sample for training
        validation_envs (int): Number of environments to sample for validation
    """
    train_envs: int
    validation_envs: int

    def _get_valid_positions(self, env) -> list[Tuple[int, int]]:
        """Get all valid positions in the grid where the agent can be placed.
        
        Args:
            env: The MiniGrid environment
            
        Returns:
            List of (x, y) coordinates where the agent can be placed
        """
        valid_positions = []
        grid = env.unwrapped.grid.encode()
        
        # Iterate through all positions in the grid
        for i in range(env.unwrapped.width):
            for j in range(env.unwrapped.height):
                # Check if the agent can overlap with the object at this position
                # .venv/lib/python3.11/site-packages/minigrid/core/world_object.py#L46
                if grid[i, j, 0] in [OBJECT_TO_IDX['empty'], OBJECT_TO_IDX['goal']]:
                    valid_positions.append((i, j))
        
        return valid_positions

    def _place_agent(self, env, pos: Tuple[int, int], dir: int = 0) -> None:
        """Place the agent at a specific position with given direction.
        
        Args:
            env: The MiniGrid environment
            pos: (x, y) position to place the agent
            dir: Direction to face (0: right, 1: down, 2: left, 3: up)
        """
        env.unwrapped.agent_pos = pos
        env.unwrapped.agent_dir = dir
        env.unwrapped.step_count = 0    # Reward, if any, is 1
        
    def _sample_pos(self, 
                        env: gym.Env, 
                        pos: tuple[int, int],
                        env_indices: list[list[int]],
                        observations_list: list[npt.NDArray],
                        actions_list: list[int],
                        rewards_list: list[float]
                        ) -> None:
        
        # For each possible direction        
        for dir in range(4):
            # Place agent at position and direction
            self._place_agent(env, pos, dir)
            
            # Get initial observation
            initial_obs = env.getObservation()
            
            # Only record the observation if it has not been recorded before
            x_idx = None
            for i in range(len(observations_list)):
                if np.array_equal(initial_obs, observations_list[i]):
                    x_idx = i
                    break
            if not x_idx:
                observations_list.append(initial_obs)
                x_idx = len(observations_list) - 1

            # For each possible action
            for action in range(env.action_space.n):
                
                # Take action and get new observation
                new_obs, reward, _terminated, _truncated, _info = env.step(action)
                
                # If the observation data has already been recorded before, skip
                y_idx = None
                for i in range(len(observations_list)):
                    if np.array_equal(new_obs, observations_list[i]):
                        y_idx = i
                        break
                if not y_idx:
                    observations_list.append(new_obs)
                    y_idx = len(observations_list) - 1
                    
                # Store the action, and reward
                actions_list.append(action)
                rewards_list.append(reward)
                
                # Store indices for this sample
                env_indices.append([x_idx, y_idx, len(actions_list)-1])
                
        return env_indices, observations_list, actions_list, rewards_list


    def custom_transform(self, data: XData) -> XData:
        """Extensively sample the environment by placing the agent at each valid position
        and trying each possible action.

        Args:
            data: XData object containing the environment
        """
        env = getattr(data, "env", None)
        if env is None:
            raise ValueError("Environment is not initialized.")

        logger.info("Extensively sampling environment")

        total_envs = self.train_envs + self.validation_envs
        observations_list = []
        actions_list = []
        rewards_list = []
        
        train_indices: list[list[int]] = []
        validation_indices: list[list[int]] = []
        
        current_env = 0
        
        while current_env < total_envs:            
            # Reset environment to get a new layout
            env.reset()
            
            # Get valid positions for this environment
            valid_positions = self._get_valid_positions(env)
            
            # Track indices for this environment's samples
            env_indices = []
            
            # For each valid position
            with tqdm(total=len(valid_positions), desc=f"Sampling Environment {current_env + 1}/{total_envs}", unit="samples") as pbar:
                for pos in valid_positions:
                    # Create samples for this position
                    self._sample_pos(
                        env, 
                        pos,
                        env_indices,
                        observations_list,
                        actions_list,
                        rewards_list
                        )

                    pbar.update(1)
            
            # Store indices in appropriate set
            if current_env < self.train_envs:
                train_indices.append(env_indices)
            else:
                validation_indices.append(env_indices)
            
            current_env += 1

        # Convert lists to numpy arrays
        data.observations = np.stack(observations_list, axis=0)
        data.actions = np.array(actions_list, dtype=np.int8).reshape(-1, 1)
        data.rewards = np.array(rewards_list, dtype=np.float16).reshape(-1, 1)
        
        # Store indices
        data.train_indices = np.array(train_indices)
        data.validation_indices = np.array(validation_indices)

        env.close()
        return data