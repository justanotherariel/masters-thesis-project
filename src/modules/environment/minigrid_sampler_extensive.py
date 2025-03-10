import gymnasium as gym
import minigrid
import minigrid.core
import minigrid.core.grid
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

logger = Logger()


class MinigridSamplerExtensive(TransformationBlock):
    """Block to extensively sample a MiniGrid Environment by placing the agent at each valid position and executing
    each possible action. Requires the environment to be a MiniGrid environment and that the last wrapper provides
    the function getObservation(), which returns the current observation without taking a step.

    Args:
        train_envs (int): Number of environments to sample for training
        validation_envs (int): Number of environments to sample for validation
    """

    train_envs: int
    train_keep_perc: float
    validation_envs: int

    def __init__(self, train_envs: int, validation_envs: int, train_keep_perc: float = 1.0):
        if train_keep_perc < 0.0 or train_keep_perc > 1.0:
            raise ValueError("train_keep_perc must be between 0.0 and 1.0")
        
        self.train_envs = train_envs
        self.train_keep_perc = train_keep_perc
        self.validation_envs = validation_envs

    def __repr__(self):
        args_to_show = [self.train_envs, self.train_keep_perc, self.validation_envs]
        args = ", ".join([f"{arg}" for arg in args_to_show])
        return f"{self.__class__.__name__}({args})"

    def _get_valid_positions(self, env) -> list[tuple[int, int]]:
        """Get all valid positions in the grid where the agent can be placed.

        Args:
            env: The MiniGrid environment

        Returns:
            List of (x, y) coordinates where the agent can be placed
        """
        valid_positions = []
        grid = env.unwrapped.grid

        # Iterate through all positions in the grid
        for y in range(grid.height):
            for x in range(grid.width):
                field = grid.get(x, y)
                # TODO: Doors will need to be treated differently
                if field is None or field.can_overlap():
                    valid_positions.append((x, y))

        return valid_positions

    def _place_agent(self, env, pos: tuple[int, int], dir: int = 0) -> None:
        """Place the agent at a specific position with given direction.

        Args:
            env: The MiniGrid environment
            pos: (x, y) position to place the agent
            dir: Direction to face (0: right, 1: down, 2: left, 3: up)
        """
        env.unwrapped.agent_pos = np.array(pos)
        env.unwrapped.agent_dir = dir
        env.unwrapped.step_count = 0  # Reward, should be 1

    def _sample_pos(
        self,
        env: gym.Env,
        pos: tuple[int, int],
        env_indices: list[list[int]],
        observations_list: list[npt.NDArray],
        actions_list: list[int],
        rewards_list: list[float],
    ) -> None:
        # For each possible direction
        for dir in range(4):
            # Place agent at position and direction
            self._place_agent(env, pos, dir)

            # Get initial observation
            initial_obs = env.getObservation()

            # Only record the observation if it has not been recorded before
            x_idx = None
            for i in range(min(len(observations_list), len(env_indices))):
                obs_idx = max(len(observations_list) - len(env_indices), 0) + i
                if np.array_equal(initial_obs, observations_list[obs_idx]):
                    x_idx = obs_idx
                    self._deduplicated_observations += 1
                    break
            if not x_idx:
                observations_list.append(initial_obs)
                x_idx = len(observations_list) - 1

            # For each possible action
            for action in range(self._info.data_info["action_space"].n.item()):
                # Place agent at position and direction
                self._place_agent(env, pos, dir)

                # Take action and get new observation
                new_obs, reward, _terminated, _truncated, _info = env.step(action)

                # If the observation data has already been recorded before, skip
                y_idx = None
                for i in range(min(len(observations_list), len(env_indices))):
                    obs_idx = max(len(observations_list) - len(env_indices), 0) + i
                    if np.array_equal(new_obs, observations_list[obs_idx]):
                        y_idx = obs_idx
                        self._deduplicated_observations += 1
                        break
                if not y_idx:
                    observations_list.append(new_obs)
                    y_idx = len(observations_list) - 1

                # Store the action, and reward
                actions_list.append(action)
                rewards_list.append(reward)

                # Store indices for this sample
                env_indices.append([x_idx, y_idx, len(actions_list) - 1])

    def custom_transform(self, data: PipelineData) -> PipelineData:
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
        train_grids: list[minigrid.core.grid.Grid] = []
        validation_indices: list[list[int]] = []
        validation_grids: list[minigrid.core.grid.Grid] = []

        current_env = 0

        self._deduplicated_observations = 0

        while current_env < total_envs:
            # Reset environment to get a new layout
            env.reset()

            # Get valid positions for this environment
            valid_positions = self._get_valid_positions(env)

            # Track indices for this environment's samples
            env_indices = []

            # For each valid position
            with tqdm(
                total=len(valid_positions), desc=f"Sampling Environment {current_env + 1}/{total_envs}", unit="samples"
            ) as pbar:
                for pos in valid_positions:
                    # Create samples for this position
                    self._sample_pos(env, pos, env_indices, observations_list, actions_list, rewards_list)
                    pbar.update(1)
                    
            if current_env < self.train_envs and self.train_keep_perc < 1.0:
                env_indices_discard_idx = np.random.choice(
                    len(env_indices), int(len(env_indices) * (1 - self.train_keep_perc)), replace=False
                )
                env_indices = [env_indices[i] for i in range(len(env_indices)) if i not in env_indices_discard_idx]

            # Store indices in appropriate set
            if current_env < self.train_envs:
                train_indices.append(env_indices)
                train_grids.append(env.unwrapped.grid)
            else:
                validation_indices.append(env_indices)
                validation_grids.append(env.unwrapped.grid)

            current_env += 1

        # Store raw sampled data
        data.observations = np.stack(observations_list, axis=0)
        data.actions = np.array(actions_list, dtype=np.int8).reshape(-1, 1)
        data.rewards = np.array(rewards_list, dtype=np.float16).reshape(-1, 1)

        # Store Metadata
        data.indices = {
            DatasetGroup.TRAIN: [np.array(env_indices) for env_indices in train_indices],
            DatasetGroup.VALIDATION: [np.array(env_indices) for env_indices in validation_indices],
            DatasetGroup.ALL: [np.array(env_indices) for env_indices in train_indices + validation_indices],
        }
        data.grids = {
            DatasetGroup.TRAIN: train_grids,
            DatasetGroup.VALIDATION: validation_grids,
            DatasetGroup.ALL: train_grids + validation_grids,
        }

        len_indices = sum([ind.shape[0] for ind in data.indices[DatasetGroup.ALL]]) * 2
        logger.info(
            f"Deduplicated {self._deduplicated_observations} observations "
            f"out of {len_indices} ({100 * self._deduplicated_observations / len_indices:.1f}%)."
        )

        env.close()
        return data

    def setup(self, info: PipelineInfo):
        self._info = info
        return info
