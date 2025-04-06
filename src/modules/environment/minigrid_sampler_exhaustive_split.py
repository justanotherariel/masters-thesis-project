import minigrid
import minigrid.core
import minigrid.core.grid
import numpy as np
from tqdm import tqdm

from src.framework.logging import Logger
from src.typing.pipeline_objects import DatasetGroup, PipelineData

from .minigrid_sampler_exhaustive import MinigridSamplerExhaustive

logger = Logger()


class MinigridSamplerExhaustiveSplit(MinigridSamplerExhaustive):
    envs: int
    perc_train: float

    def __init__(self, envs: int, perc_train: float):
        self.envs = envs
        self.perc_train = perc_train

    def __repr__(self):
        args_to_show = [self.envs, self.perc_train]
        args = ", ".join([f"{arg}" for arg in args_to_show])
        return f"{self.__class__.__name__}({args})"

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

        observations_list = []
        actions_list = []
        rewards_list = []

        train_indices: list[list[int]] = []
        train_grids: list[minigrid.core.grid.Grid] = []
        validation_indices: list[list[int]] = []
        validation_grids: list[minigrid.core.grid.Grid] = []

        self._deduplicated_observations = 0

        for env_idx in range(self.envs):
            # Reset environment to get a new layout
            env.reset()

            # Get valid positions for this environment
            valid_positions = self._get_valid_positions(env)

            # Track indices for this environment's samples
            env_indices = []

            # For each valid position
            with tqdm(
                total=len(valid_positions), desc=f"Sampling Environment {env_idx + 1}/{self.envs}", unit="samples"
            ) as pbar:
                for pos in valid_positions:
                    # Create samples for this position
                    self._sample_pos(env, pos, env_indices, observations_list, actions_list, rewards_list)
                    pbar.update(1)

            # Randomly split the indices into train and validation
            train_indices_idx = np.random.choice(
                len(env_indices), int(self.perc_train * len(env_indices)), replace=False
            )
            train_indices_idx = np.sort(train_indices_idx)

            train_indices.append([env_indices[i] for i in train_indices_idx])
            validation_indices.append([env_indices[i] for i in range(len(env_indices)) if i not in train_indices_idx])

            train_grids.append(env.unwrapped.grid)
            validation_grids.append(env.unwrapped.grid)

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
