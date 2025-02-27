import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from minigrid.core import actions
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tqdm import tqdm
import math

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.accuracy import obs_argmax
from src.modules.training.datasets.simple import SimpleDatasetDefault
from src.modules.training.datasets.tensor_index import TensorIndex
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .data_transform import dataset_to_list

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
            self.create_sample_eval_pdf(
                data,
                dataset_group,
                constrain_to_one_agent=self.constrain_to_one_agent,
                prefix="constrained" if self.constrain_to_one_agent else "",
            )

        logger.info("Sample evaluation (image) complete.")
        return data


    def create_sample_eval_pdf(
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
        pred_obs, pred_reward = data.predictions[dataset_group][:2]
        pred_ti = self.info.model_ti
        
        eval_n_grids = min(self.eval_n_grids, len(indices)) if self.eval_n_grids is not None else len(indices)

        # Argmax the predictions
        pred_obs = obs_argmax(pred_obs, pred_ti, constrain_to_one_agent=constrain_to_one_agent)

        # Go through each grid
        grid_idx_start = 0
        for grid_idx in tqdm(range(eval_n_grids), desc=f"Dataset {dataset_group.name.lower()}"):
            with PDFFileWriter(self.info.output_dir, f"sample_eval_{prefix}{dataset_group.name.lower()}_{grid_idx}.pdf") as writer:
                grid_index_len = len(indices[grid_idx])
                create_sample_eval_pdf(
                    x_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    x_action[grid_idx_start : grid_idx_start + grid_index_len],
                    y_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    y_reward[grid_idx_start : grid_idx_start + grid_index_len],
                    pred_obs[grid_idx_start : grid_idx_start + grid_index_len],
                    pred_reward[grid_idx_start : grid_idx_start + grid_index_len],
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
            continue    # Skip correct predictions
        
        # Get the agent positions for x, y, and pred observations
        agent_x_obs_pos = (x_obs[sample_idx, :, :, 3].squeeze() != 0).nonzero()[0]
        agent_x_obs_dir = (
            x_obs[sample_idx, agent_x_obs_pos[0], agent_x_obs_pos[1], 3].item() - 1
        )
        agent_x_obs_pos = [((agent_x_obs_pos[0].item(), agent_x_obs_pos[1].item()), agent_x_obs_dir)]

        agent_y_obs_pos = (y_obs[sample_idx, :, :, 3].squeeze() != 0).nonzero()[0]
        agent_y_obs_dir = (
            y_obs[sample_idx, agent_y_obs_pos[0], agent_y_obs_pos[1], 3].item() - 1
        )
        agent_y_obs_pos = [((agent_y_obs_pos[0].item(), agent_y_obs_pos[1].item()), agent_y_obs_dir)]

        agent_pred_obs_pos_tmp = (
            pred_obs[sample_idx, :, :, 3].squeeze() != 0
        ).nonzero()
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
        writer.add_row(x_obs_img, ACTION_STR[action], y_obs_img, y_reward_val, pred_obs_img, pred_reward_val, obs_correct, reward_correct)


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


class PDFFileWriter:
    def __init__(self, dir: Path, filename: str):
        self.file_path = dir / filename
        self.row_height = 100
        self.page_width, self.page_height = A4
        self.margin = 30

        # Calculate dimensions
        self.usable_width = self.page_width - (2 * self.margin)
        self.image_width = self.usable_width / 6  # Images take up 1/6 of usable width
        self.image_height = self.row_height - 15  # Slightly smaller than row height

        # Initialize PDF
        self.c = canvas.Canvas(str(self.file_path), pagesize=A4)
        self.current_y = self.page_height - self.margin

        # Add arrow symbol
        self.arrow_width = 30

    def _add_arrow(self, x, y, color: str):
        """Draw an arrow symbol"""
        self.c.setStrokeColor(color)
        self.c.setLineWidth(2)
        # Draw arrow line
        self.c.line(x, y, x + self.arrow_width, y)
        # Draw arrow head
        self.c.line(x + self.arrow_width - 10, y + 5, x + self.arrow_width, y)
        self.c.line(x + self.arrow_width - 10, y - 5, x + self.arrow_width, y)

    def _add_vertical_separator(self, x, y, height):
        """Draw a vertical separator line"""
        self.c.setStrokeColor("black")
        self.c.setLineWidth(1)
        self.c.line(x, y, x, y + height)

    def _format_reward(self, reward: float) -> str:
        """Format reward value with sign and fixed decimal places"""
        sign = "+" if reward >= 0 else "-"
        return f"{sign} {abs(reward):.2f}"

    def add_row(self, x_obs: Image.Image, x_action: str, y_obs: Image.Image, y_reward: float, pred_obs: Image.Image, pred_reward: float, correct_obs: bool, correct_reward: bool):
        """
        Add a row with observation images, action text, and rewards.

        Args:
            x_obs: Input observation (PIL Image)
            x_action: Action index
            y_obs: Target observation (PIL Image)
            y_reward: Target reward value
            pred_obs: Predicted observation (PIL Image)
            pred_reward: Predicted reward value
            correct: Whether the prediction was correct
        """
        # Check if we need a new page
        if self.current_y - self.row_height < self.margin:
            self.c.showPage()
            self.current_y = self.page_height - self.margin

        # Calculate positions
        x_positions = []
        x_positions.append(self.margin)  # x_obs
        x_positions.append(x_positions[-1] + self.image_width)  # action text
        x_positions.append(x_positions[-1] + self.image_width)  # arrow
        x_positions.append(x_positions[-1] + 50)  # rewards
        x_positions.append(x_positions[-1] + self.image_width)  # y_obs
        x_positions.append(x_positions[-1] + self.image_width + 20)  # separator
        x_positions.append(x_positions[-1] + 20)  # pred_obs

        # Draw images and elements
        for img, pos in [(x_obs, 0), (y_obs, 4), (pred_obs, 6)]:
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            self.c.drawImage(
                ImageReader(img_buffer),
                x_positions[pos],
                self.current_y - self.image_height,
                width=self.image_width,
                height=self.image_height,
                preserveAspectRatio=True,
            )

        # Add action text
        self.c.setFont("Helvetica", 12)
        text_width = self.c.stringWidth(x_action, "Helvetica", 12)
        text_x = x_positions[1] + (self.image_width - text_width) / 2
        text_y = self.current_y - self.image_height / 2
        self.c.drawString(text_x, text_y, x_action)

        # Add arrow
        arrow_color = "green" if (correct_obs and correct_reward) else "red"
        self._add_arrow(x_positions[2], self.current_y - self.image_height / 2, arrow_color)

        # Add reward values
        self.c.setFont("Courier", 9)  # Use monospace font for aligned numbers
        self.c.setFillColor("black" if correct_reward else "red")
        reward_x = x_positions[3]
        # Target reward (top)
        target_text = f"Target: {self._format_reward(y_reward)}"
        self.c.drawString(reward_x, self.current_y - self.image_height / 3, target_text)
        # Predicted reward (bottom)
        pred_text = f"Pred  : {self._format_reward(pred_reward)}"
        self.c.drawString(reward_x, self.current_y - self.image_height * 2/3, pred_text)
        self.c.setFillColor("black")

        # Add vertical separator
        self._add_vertical_separator(x_positions[5], self.current_y - self.image_height, self.image_height)

        # Update current_y position
        self.current_y -= self.row_height

    def add_page_break(self):
        """Add a page break and reset the y-position to the top of the new page."""
        self.c.showPage()
        self.current_y = self.page_height - self.margin

    def close(self):
        """Save and close the PDF."""
        self.c.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()