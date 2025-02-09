import io
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

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.simple import SimpleDatasetDefault
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .data_transform import dataset_to_list

logger = Logger()

ACTION_STR = [actions.Actions(i).name for i in range(7)]


class MinigridSampleEvalImage(TransformationBlock):
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
        logger.info("Evaluating samples (image)...")

        for dataset_group in data.grids:
            if dataset_group == DatasetGroup.ALL:
                continue
            find_errors(data, self.info, dataset_group, self.info.output_dir)

        logger.info("Sample evaluation (image) complete.")
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
    pred_obs, pred_reward = data.predictions[dataset_group]
    pred_ti = info.model_ti

    # Argmax the predictions
    pred_obs_argmax = torch.empty_like(x_obs)
    for obs_idx in range(len(pred_ti.observation)):
        pred_obs_argmax[..., obs_idx] = torch.argmax(pred_obs[..., pred_ti.observation[obs_idx]], dim=3)
    pred_obs = pred_obs_argmax

    # Go through each grid
    grid_idx_start = 0
    # for grid_idx in tqdm(range(len(indices)), desc=f"Dataset {dataset_group.name.lower()}"):
    for grid_idx in tqdm(range(1), desc=f"Dataset {dataset_group.name.lower()}"):
        with PDFFileWriter(output_dir, f"sample_eval_{dataset_group.name.lower()}_{grid_idx}.pdf") as writer:
            grid_index_len = len(indices[grid_idx])
            x_action_grid = x_action[grid_idx_start : grid_idx_start + grid_index_len]
            x_obs_grid = x_obs[grid_idx_start : grid_idx_start + grid_index_len]
            y_obs_grid = y_obs[grid_idx_start : grid_idx_start + grid_index_len]
            pred_obs_grid = pred_obs[grid_idx_start : grid_idx_start + grid_index_len]

            # Go through each sample
            for sample_idx in range(grid_index_len):
                agent_x_obs_pos = (x_obs_grid[sample_idx, :, :, target_ti.observation[3]].squeeze() != 0).nonzero()[0]
                agent_x_obs_dir = (
                    x_obs_grid[sample_idx, agent_x_obs_pos[0], agent_x_obs_pos[1], target_ti.observation[3]].item() - 1
                )
                agent_x_obs_pos = [((agent_x_obs_pos[0].item(), agent_x_obs_pos[1].item()), agent_x_obs_dir)]

                agent_y_obs_pos = (y_obs_grid[sample_idx, :, :, target_ti.observation[3]].squeeze() != 0).nonzero()[0]
                agent_y_obs_dir = (
                    y_obs_grid[sample_idx, agent_y_obs_pos[0], agent_y_obs_pos[1], target_ti.observation[3]].item() - 1
                )
                agent_y_obs_pos = [((agent_y_obs_pos[0].item(), agent_y_obs_pos[1].item()), agent_y_obs_dir)]

                agent_pred_obs_pos_tmp = (
                    pred_obs_grid[sample_idx, :, :, target_ti.observation[3]].squeeze() != 0
                ).nonzero()
                agent_pred_obs_pos = []
                for i in range(agent_pred_obs_pos_tmp.shape[0]):
                    x = agent_pred_obs_pos_tmp[i, 0].item()
                    y = agent_pred_obs_pos_tmp[i, 1].item()
                    agent_pred_obs_dir = pred_obs_grid[sample_idx, x, y, target_ti.observation[3]].item() - 1

                    agent_pred_obs_pos.append(((x, y), agent_pred_obs_dir))

                action = x_action_grid[sample_idx].item()

                x_obs_img = render_grid(x_obs_grid[sample_idx], agent_x_obs_pos)
                y_obs_img = render_grid(y_obs_grid[sample_idx], agent_y_obs_pos)
                pred_obs_img = render_grid(pred_obs_grid[sample_idx], agent_pred_obs_pos)

                prediction_correct = (y_obs_grid[sample_idx] == pred_obs_grid[sample_idx]).all()

                writer.add_row(x_obs_img, ACTION_STR[action], y_obs_img, pred_obs_img, prediction_correct)

            grid_idx_start += grid_index_len


def render_grid(obs: torch.Tensor, agents: list[tuple[tuple[int, int], int]]):
    tile_size = 32

    grid = Grid(obs.shape[0], obs.shape[1])
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            obj = obs[i, j, 0].item()
            color = obs[i, j, 1].item()
            state = obs[i, j, 2].item()
            world_obj = WorldObj.decode(obj, color, state)
            grid.set(i, j, world_obj)

    width_px = grid.width * tile_size
    height_px = grid.height * tile_size

    img_np = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for j in range(0, grid.height):
        for i in range(0, grid.width):
            cell = grid.get(i, j)

            agent_here = False
            for agent in agents:
                if np.array_equal(agent[0], (i, j)):
                    agent_here = True
                    agent_dir = agent[1]
                    break

            tile_img = Grid.render_tile(
                cell,
                agent_dir=agent_dir if agent_here else None,
                highlight=False,
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

    def add_row(self, x_obs: Image.Image, x_action: str, y_obs: Image.Image, pred_obs: Image.Image, correct: bool):
        """
        Add a row with observation images and action text.

        Args:
            x_obs: Input observation (PIL Image)
            x_action: Action index
            y_obs: Target observation (PIL Image)
            pred_obs: Predicted observation (PIL Image)
        """
        # Check if we need a new page
        if self.current_y - self.row_height < self.margin:
            self.c.showPage()
            self.current_y = self.page_height - self.margin

        # Calculate positions
        x_positions = [
            self.margin,  # x_obs
            self.margin + self.image_width + 20,  # action text
            self.margin + (self.image_width + 20) * 2,  # arrow
            self.margin + (self.image_width + 20) * 3,  # y_obs
            self.margin + (self.image_width + 20) * 4,  # separator
            self.margin + (self.image_width + 20) * 4 + 20,  # pred_obs
        ]

        # Draw images and elements
        for img, pos in [(x_obs, 0), (y_obs, 3), (pred_obs, 5)]:
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
        arrow_color = "green" if correct else "red"
        self._add_arrow(x_positions[2], self.current_y - self.image_height / 2, arrow_color)

        # Add vertical separator
        self._add_vertical_separator(x_positions[4], self.current_y - self.image_height, self.image_height)

        # Update current_y position
        self.current_y -= self.row_height

    def close(self):
        """Save and close the PDF."""
        self.c.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
