import math
from dataclasses import dataclass

import numpy as np
import torch
from minigrid.core import actions
from minigrid.core.constants import DIR_TO_VEC
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
class MinigridAttentionEval(TransformationBlock):
    """Score the predictions of the model."""

    eval_n_correct: int = 5
    eval_n_incorrect: int = 5

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
        logger.info("Evaluating samples (attention map)...")

        for dataset_group in data.grids:
            if dataset_group == DatasetGroup.ALL:
                continue

            if len(data.predictions[dataset_group]) < 3:
                logger.info(
                    f"Skipping sample evaluation (attention) because predictions "
                    f"attention maps are not available. ({dataset_group.name})"
                )
                continue

            self.create_sample_eval_pdf(data, dataset_group)

        logger.info("Sample evaluation (image) complete.")
        return data

    def create_sample_eval_pdf(
        self,
        data: PipelineData,
        dataset_group: DatasetGroup,
    ):
        """Calculate the accuracy of the model.

        :param index_name: The name of the indice.
        """
        target_data = dataset_to_list(data, dataset_group)
        x_obs, x_action = target_data[0]
        y_obs, y_reward = target_data[1]
        pred_obs, pred_reward, pred_eta = data.predictions[dataset_group]
        pred_ti = self.info.model_ti

        # Argmax the predictions
        pred_obs = obs_argmax(pred_obs, pred_ti)

        # Go through each grid
        progress_bar = tqdm(
            total=self.eval_n_correct + self.eval_n_incorrect, desc=f"Creating PDF ({dataset_group.name})"
        )
        writer = PDFFileWriter(self.info.output_dir, f"attention_eval_{dataset_group.name.lower()}.pdf")

        create_sample_eval_pdf(
            x_obs,
            x_action,
            y_obs,
            y_reward,
            pred_obs,
            pred_reward,
            pred_eta,
            writer=writer,
            progress_bar=progress_bar,
            n_correct=self.eval_n_correct,
            n_incorrect=self.eval_n_incorrect,
        )
        writer.close()
        progress_bar.close()


def create_sample_eval_pdf(
    x_obs: torch.Tensor,
    x_action: torch.Tensor,
    y_obs: torch.Tensor,
    y_reward: torch.Tensor,
    pred_obs: torch.Tensor,
    pred_reward: torch.Tensor,
    pred_eta: torch.Tensor,
    writer: "PDFFileWriter",
    progress_bar: tqdm,
    n_correct: int,
    n_incorrect: int,
):
    """Calculate the accuracy of the model.

    :param index_name: The name of the indice.
    """
    found_correct = 0
    found_incorrect = 0

    # Randomize the samples
    indices = torch.randperm(len(x_obs))

    # Go through each sample
    for sample_idx in indices:
        if found_correct >= n_correct and found_incorrect >= n_incorrect:
            break

        # Only show samples with action 2 (forward)
        # if x_action[sample_idx] != 2:
        #     continue

        # Get the reward
        y_reward_val = y_reward[sample_idx].item()
        pred_reward_val = pred_reward[sample_idx].item()

        # Check if the prediction was correct
        obs_correct = (y_obs[sample_idx] == pred_obs[sample_idx]).all().item()
        reward_correct = math.isclose(y_reward_val, pred_reward_val, abs_tol=0.2)
        prediction_correct = obs_correct and reward_correct

        if found_correct >= n_correct and prediction_correct:
            continue
        elif prediction_correct:
            found_correct += 1

        if found_incorrect >= n_incorrect and not prediction_correct:
            continue
        elif not prediction_correct:
            found_incorrect += 1

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
            sample_idx,
            x_obs_img,
            ACTION_STR[action],
            y_obs_img,
            y_reward_val,
            pred_obs_img,
            pred_reward_val,
            obs_correct,
            reward_correct,
        )

        # Softmax the attention map
        # eta = torch.nn.functional.softmax(pred_eta[sample_idx], dim=1)

        # Normalize the attention map
        # eta = pred_eta[sample_idx] / pred_eta[sample_idx].max(dim=1, keepdim=True).values
        row_max_values, _ = torch.max(pred_eta[sample_idx], dim=-1, keepdim=True)
        safe_max_values = torch.clamp(row_max_values, min=1e-10)
        eta = pred_eta[sample_idx] / safe_max_values

        # Highlight feature and target agent positions
        current_pos_idx = agent_x_obs_pos[0][0][0] * x_obs.shape[1] + agent_x_obs_pos[0][0][1]
        x_hightlight = (current_pos_idx, "green", "green")

        direction = DIR_TO_VEC[agent_x_obs_dir]
        forward_pos_idx = (
            (agent_x_obs_pos[0][0][0] + direction[0]) * y_obs.shape[1] + agent_x_obs_pos[0][0][1] + direction[1]
        )
        forward_hightlight = (forward_pos_idx, "red", "red")

        writer.add_tensor(eta, [x_hightlight, forward_hightlight])
        writer.add_page_break()

        progress_bar.update(1)


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
