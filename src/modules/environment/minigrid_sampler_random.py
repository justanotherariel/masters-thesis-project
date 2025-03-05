from dataclasses import dataclass

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


@dataclass
class MinigridSamplerRandom(TransformationBlock):
    """Block to sample a Gymnasium Environment.

    :param environment: The environment to instantiate. E.g. MiniGrid-Empty-5x5-v0
    """

    num_samples: int
    num_samples_per_env: int
    perc_train: float

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._info = info
        return info

    def custom_transform(self, data: PipelineData) -> npt.NDArray[np.float32]:
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
        train_grids: list[minigrid.core.grid.Grid] = []
        validation_indices: list[list[int]] = []
        validation_grids: list[minigrid.core.grid.Grid] = []

        samples_idx = 0

        progress_bar = tqdm(total=self.num_samples, desc="Sampling Environment", unit="samples")
        while samples_idx < self.num_samples:
            trajectory: list[int] = []
            observation, _info = env.reset()

            for _ in range(self.num_samples_per_env):
                # Save current State
                observations.append(observation)

                # Take a step
                action = np.random.randint(0, self._info.data_info["action_space"].n.item())
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
                train_grids.append(env.unwrapped.grid)
            else:
                validation_indices.append(indices_collected)
                validation_grids.append(env.unwrapped.grid)

        # Store raw sampled data
        data.observations = np.stack(observations, axis=0)

        # Store Metadata
        data.indices = {
            DatasetGroup.TRAIN: np.stack(train_indices, axis=0),
            DatasetGroup.VALIDATION: np.stack(validation_indices, axis=0),
            DatasetGroup.ALL: np.stack(train_indices + validation_indices, axis=0),
        }
        data.grids = {
            DatasetGroup.TRAIN: train_grids,
            DatasetGroup.VALIDATION: validation_grids,
            DatasetGroup.ALL: train_grids + validation_grids,
        }

        env.close()
        return data
