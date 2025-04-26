import math
from dataclasses import dataclass

import numpy as np
import torch
from minigrid.core import actions
from minigrid.core.grid import Grid
from PIL import Image
from tqdm import tqdm

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.accuracy import obs_argmax
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .data_transform import dataset_to_list
from .pdf_file_writer import PDFFileWriter

logger = Logger()

ACTION_STR = [actions.Actions(i).name for i in range(7)]


@dataclass
class MinigridSampleEvalImage(TransformationBlock):
    """Score the predictions of the model."""

    eval_n_grids: int | None = None
    constrain_to_one_agent: bool = False
    only_errors: bool = False

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
        logger.info("Evaluating samples (image)...")

        for dataset_group in data.grids:
            if dataset_group == DatasetGroup.ALL:
                continue
            self._create_sample_eval_pdf(
                data,
                dataset_group,
                constrain_to_one_agent=self.constrain_to_one_agent,
                prefix="constrained" if self.constrain_to_one_agent else "",
            )

        logger.info("Sample evaluation (image) complete.")
        return data

    def _create_sample_eval_pdf(
        self,
        data: PipelineData,
        dataset_group: DatasetGroup,
        *,
        prefix: str = "",
        constrain_to_one_agent: bool = False,
    ):
        """Calculate the accuracy of the model.

        :param index_name: The name of the indice.
        """
        prefix = f"{prefix}_" if prefix else ""

        indices = data.indices[dataset_group]
        target_data = dataset_to_list(data, dataset_group)
        x_obs, x_action = target_data[0]
        y_obs, y_reward = target_data[1]
        pred_obs, pred_reward = data.predictions[dataset_group]['pred_obs'], data.predictions[dataset_group]['pred_reward']
        pred_ti = self.info.model_ti

        eval_n_grids = min(self.eval_n_grids, len(indices)) if self.eval_n_grids is not None else len(indices)

        # Argmax the predictions
        pred_obs = obs_argmax(pred_obs, pred_ti, constrain_to_one_agent=constrain_to_one_agent)

        # Go through each grid
        grid_idx_start = 0
        for grid_idx in tqdm(range(eval_n_grids), desc=f"Dataset {dataset_group.name.lower()}"):
            with PDFFileWriter(
                self.info.output_dir, f"sample_eval_{prefix}{dataset_group.name.lower()}_{grid_idx}.pdf"
            ) as writer:
                grid_index_len = len(indices[grid_idx])
                create_sample_eval_pdf(
                    x_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    x_action[grid_idx_start : grid_idx_start + grid_index_len],
                    y_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    y_reward[grid_idx_start : grid_idx_start + grid_index_len],
                    pred_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    pred_reward[grid_idx_start : grid_idx_start + grid_index_len],
                    start_idx=grid_idx_start,
                    writer=writer,
                    errors_only=True,
                )

                if not self.only_errors:
                    writer.add_page_break()
                    writer.add_page_break()

                    create_sample_eval_pdf(
                        x_obs[grid_idx_start : grid_idx_start + grid_index_len],
                        x_action[grid_idx_start : grid_idx_start + grid_index_len],
                        y_obs[grid_idx_start : grid_idx_start + grid_index_len],
                        y_reward[grid_idx_start : grid_idx_start + grid_index_len],
                        pred_obs[grid_idx_start : grid_idx_start + grid_index_len],
                        pred_reward[grid_idx_start : grid_idx_start + grid_index_len],
                        start_idx=grid_idx_start,
                        writer=writer,
                        errors_only=False,
                    )

            grid_idx_start += grid_index_len


def create_sample_eval_pdf(
    x_obs: torch.Tensor,
    x_action: torch.Tensor,
    y_obs: torch.Tensor,
    y_reward: torch.Tensor,
    pred_obs: torch.Tensor,
    pred_reward: torch.Tensor,
    start_idx: int,
    writer: "PDFFileWriter",
    *,
    errors_only: bool = False,
):
    """Calculate the accuracy of the model.

    :param index_name: The name of the indice.
    """
    # Go through each sample
    for sample_idx in range(len(x_obs)):
        # Get the reward
        y_reward_val = y_reward[sample_idx].item()
        pred_reward_val = pred_reward[sample_idx].item()

        # Check if the prediction was correct
        obs_correct = (y_obs[sample_idx] == pred_obs[sample_idx]).all().item()
        reward_correct = math.isclose(y_reward_val, pred_reward_val, abs_tol=0.2)
        prediction_correct = obs_correct and reward_correct
        if errors_only and prediction_correct:
            continue  # Skip correct predictions

        # Get the agent positions for x, y, and pred observations
        agent_x_obs_pos = (x_obs[sample_idx, :, :, 3].squeeze() != 0).nonzero()[0]
        agent_x_obs_dir = x_obs[sample_idx, agent_x_obs_pos[0], agent_x_obs_pos[1], 3].item() - 1
        agent_x_obs_pos = [((agent_x_obs_pos[0].item(), agent_x_obs_pos[1].item()), agent_x_obs_dir)]

        agent_y_obs_pos = (y_obs[sample_idx, :, :, 3].squeeze() != 0).nonzero()[0]
        agent_y_obs_dir = y_obs[sample_idx, agent_y_obs_pos[0], agent_y_obs_pos[1], 3].item() - 1
        agent_y_obs_pos = [((agent_y_obs_pos[0].item(), agent_y_obs_pos[1].item()), agent_y_obs_dir)]

        agent_pred_obs_pos_tmp = (pred_obs[sample_idx, :, :, 3].squeeze() != 0).nonzero()
        agent_pred_obs_pos = []
        for i in range(agent_pred_obs_pos_tmp.shape[0]):
            x = agent_pred_obs_pos_tmp[i, 0].item()
            y = agent_pred_obs_pos_tmp[i, 1].item()
            agent_pred_obs_dir = pred_obs[sample_idx, x, y, 3].item() - 1

            agent_pred_obs_pos.append(((x, y), agent_pred_obs_dir))

        # Render the images for x, y, and pred observations
        x_obs_img = render_grid(x_obs[sample_idx], None, agent_x_obs_pos)
        y_obs_img = render_grid(y_obs[sample_idx], None, agent_y_obs_pos)
        pred_obs_img = render_grid(pred_obs[sample_idx], y_obs[sample_idx], agent_pred_obs_pos)

        # Get the action
        action = x_action[sample_idx].item()

        # Add the row to the PDF
        writer.add_row(
            start_idx + sample_idx,
            x_obs_img,
            ACTION_STR[action],
            y_obs_img,
            y_reward_val,
            pred_obs_img,
            pred_reward_val,
            obs_correct,
            reward_correct,
        )


def render_grid(obs: torch.Tensor, target: torch.Tensor | None, agents: list[tuple[tuple[int, int], int]]):
    tile_size = 32

    grid = Grid.decode(obs.numpy()[..., :3])[0]
    width_px = grid.width * tile_size
    height_px = grid.height * tile_size

    error_mask = (obs != target).any(dim=-1).numpy() if target is not None else None

    # Render the grid
    img_np = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
    for j in range(0, grid.height):
        for i in range(0, grid.width):
            cell = grid.get(i, j)

            agent_dir = None
            for agent in agents:
                if np.array_equal(agent[0], (i, j)):
                    agent_dir = agent[1]
                    break

            tile_img = Grid.render_tile(
                cell,
                agent_dir=agent_dir,
                highlight=error_mask[i, j] if error_mask is not None else False,
                tile_size=tile_size,
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img_np[ymin:ymax, xmin:xmax, :] = tile_img

    return Image.fromarray(img_np)
