import io
from pathlib import Path

import torch
from minigrid.core import actions
from tqdm import tqdm

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.simple import SimpleDatasetDefault
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .data_transform import dataset_to_list

logger = Logger()

ERROR_TYPES = ["object", "color", "state", "agent"]
DIR_STR = ["none", "right", "down", "left", "up"]
ACTION_STR = [actions.Actions(i).name for i in range(7)]


class MinigridSampleEvalText(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        return info

    def custom_transform(self, data: PipelineData, **kwargs) -> PipelineData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        logger.info("Evaluating samples (text)...")

        for dataset_group in data.grids:
            if dataset_group == DatasetGroup.ALL:
                continue
            find_errors(data, self.info, dataset_group, self.info.output_dir)

        logger.info("Sample evaluation (text) complete.")
        return data


def find_errors(
    data: PipelineData,
    info: PipelineInfo,
    dataset_group: DatasetGroup,
    output_dir: Path,
):
    """Calculate the accuracy of the model.

    :param index_name: The name of the indice.
    """
    indices = data.indices[dataset_group]
    target_data = dataset_to_list(data, dataset_group)
    x_obs, x_action = target_data[0]
    y_obs, y_reward = target_data[1]
    target_ti = SimpleDatasetDefault.create_ti(info, discrete=False)
    pred_obs, pred_reward = data.predictions[dataset_group]["pred_obs"], data.predictions[dataset_group]["pred_reward"]
    pred_ti = info.model_ti

    # Argmax the predictions
    pred_obs_argmax = torch.empty_like(x_obs)
    for obs_idx in range(len(pred_ti.observation)):
        pred_obs_argmax[..., obs_idx] = torch.argmax(pred_obs[..., pred_ti.observation[obs_idx]], dim=3)
    pred_obs = pred_obs_argmax

    # Find errors
    grid_idx_start = 0
    with TextFileWriter(output_dir, f"sample_eval_{dataset_group.name.lower()}.txt") as writer:
        # Go through each grid
        for grid_idx in tqdm(range(len(indices)), desc=f"Dataset {dataset_group.name.lower()}"):
            writer.new_grid(grid_idx)

            grid_index_len = len(indices[grid_idx])
            x_action_grid = x_action[grid_idx_start : grid_idx_start + grid_index_len]
            x_obs_grid = x_obs[grid_idx_start : grid_idx_start + grid_index_len]
            y_obs_grid = y_obs[grid_idx_start : grid_idx_start + grid_index_len]
            pred_obs_grid = pred_obs[grid_idx_start : grid_idx_start + grid_index_len]

            # Go through each sample
            for sample_idx in range(grid_index_len):
                agent_starting_pos = (x_obs_grid[sample_idx, :, :, target_ti.observation[3]].squeeze() != 0).nonzero()[
                    0
                ]
                agent_starting_dir = x_obs_grid[
                    sample_idx, agent_starting_pos[0], agent_starting_pos[1], target_ti.observation[3]
                ].item()
                action = x_action_grid[sample_idx].item()

                writer.new_sample(sample_idx, action, agent_starting_pos, agent_starting_dir)

                sample_errors = y_obs_grid[sample_idx] != pred_obs_grid[sample_idx]
                if sample_errors.any():
                    # Find the location of the error
                    error_loc = sample_errors.nonzero()

                    # Print each error
                    for error_idx in range(error_loc.shape[0]):
                        # Find the type of error (object, color, state, agent)
                        error_type = ERROR_TYPES[error_loc[error_idx, 2]]

                        # Find the true and predicted values
                        true_val = y_obs_grid[
                            sample_idx, error_loc[error_idx, 0], error_loc[error_idx, 1], error_loc[error_idx, 2]
                        ]
                        pred_val = pred_obs_grid[
                            sample_idx, error_loc[error_idx, 0], error_loc[error_idx, 1], error_loc[error_idx, 2]
                        ]

                        # Print the error
                        writer.new_error(
                            (error_loc[error_idx, 0], error_loc[error_idx, 1]), error_type, true_val, pred_val
                        )

            grid_idx_start += grid_index_len


class TextFileWriter:
    def __init__(self, dir: Path, filename: str):
        self.file_path = dir / filename
        self.buffer = io.StringIO()

        self.indent_space = 4
        self._indent = " " * self.indent_space

    def new_grid(self, grid_idx: int):
        if grid_idx != 0:
            self.buffer.write("\n\n\n\n")
        self.buffer.write(f"---------------- Grid {grid_idx} ----------------\n")

    def new_sample(self, sample_idx: int, action: int, agent_starting_pos: tuple[int, int], agent_starting_dir: int):
        if sample_idx != 0:
            self.buffer.write("\n")

        action_str = ACTION_STR[action]
        dir_str = DIR_STR[agent_starting_dir]
        self.buffer.write(
            f"Sample [{sample_idx:>4}] - ({agent_starting_pos[0]:>2}, "
            f"{agent_starting_pos[1]:>2}), {dir_str:>5} | {action_str:>7}\n"
        )

    def new_error(self, error_pos: tuple[int, int], error_type: str, true_val: int, pred_val: int):
        true_str = f"{true_val:>2}"
        pred_str = f"{pred_val:>2}"
        if error_type == "agent":
            true_str = DIR_STR[true_val]
            pred_str = DIR_STR[pred_val]

        self.buffer.write(
            f"{self._indent}{error_type} ({error_pos[0]}, {error_pos[1]}) - True: {true_str}, Predicted: {pred_str}\n"
        )

    def close(self):
        with open(self.file_path, "a") as f:
            f.write(self.buffer.getvalue())
        self.buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
