from typing import Any

import minigrid.core
import minigrid.core.grid
import torch
from PIL import Image
import numpy as np
import minigrid
from minigrid.core.grid import Grid
import wandb
from PIL import ImageDraw

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import XData
from src.modules.training.datasets.two_d_dataset import TwoDDataset

logger = Logger()

OBSERVATION_VARIABLES = [
    "object",
    "color",
    "state",
    "agent",
]

class MinigridHeatmap(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        return info

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """

        output_dir = self.info["output_dir"]
        for grid_idx, grid in enumerate(data.grids):
            grid_size = (grid.width, grid.height)

            # Create img of grid
            img = create_img(grid, tile_size=32)
            img_path = output_dir / f"grid_{grid_idx}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)

            # Get data
            dataset = TwoDDataset(data, indices="train_indices", discretize=False)
            train_input = list(dataset)
            train_predictions = data.train_predictions

            certainty = calculate_certainty(train_input, train_predictions, self.info)
            certainty_np = data_py_to_np(certainty, grid_size=grid_size)
            wandb_masks = {key: None for key in certainty_np}
            for key in certainty_np:
                overlay = create_overlay(certainty_np[key], grid_size=grid_size, tile_size=32)
                np_overlay = np.array(overlay)
                shaded_image = Image.alpha_composite(img.convert("RGBA"), overlay)
                shaded_image.save(output_dir / f"grid_{grid_idx}_{key}.png")
                
                wandb_masks[key] = {
                    "mask_data": np.where(np_overlay[:, :, 0] == 255, 1, 0)
                }

            # Log to wandb
            logger.log_to_external({f"Grid {grid_idx}": wandb.Image(img, masks=wandb_masks)})


        return data
    
def create_img(grid: Grid, tile_size: int) -> Image:
    img_np = grid.render(   
        tile_size=tile_size,
        agent_pos=(1, 1),
        agent_dir=None,
        highlight_mask=None,
    ).astype(np.uint8)
    return Image.fromarray(img_np)

def calculate_certainty(input_data, prediction_data, info):
    # Extract input_data
    x_obs, x_actions, y_obs, y_rewards = [], [], [], []
    for sample_idx in range(len(input_data)):
        x_obs.append(input_data[sample_idx][0][0])
        x_actions.append(input_data[sample_idx][0][1])
        y_obs.append(input_data[sample_idx][1][0])
        y_rewards.append(input_data[sample_idx][1][1])
    x_obs = torch.stack(x_obs)
    x_actions = torch.stack(x_actions)
    y_obs = torch.stack(y_obs)
    y_rewards = torch.stack(y_rewards)
    input_ti = TwoDDataset.create_ti(info)

    # Extract prediction_data
    pred_obs, pred_rewards = prediction_data
    pred_ti = info["token_index"]
    pred_ti.discrete = True

    # Check that all lengths are the same
    assert len(pred_obs) == len(pred_rewards) == len(x_obs) == len(x_actions) == len(y_obs) == len(y_rewards)
    n_samples = len(pred_obs)

    # Calculate the accuarcy
    data: dict[str, dict[tuple[int, int], list[float]]] = {
        key: {} for key in OBSERVATION_VARIABLES
    }
    for sample_idx in range(n_samples):

        # Get the agent's position
        agent_x_pos = torch.nonzero(
            x_obs[sample_idx, :, :, input_ti.observation[3]] > 0
        ).tolist()
        assert len(agent_x_pos) == 1
        agent_x_pos = tuple(agent_x_pos[0][:-1])

        for key in data:
            if agent_x_pos not in data[key]:
                data[key][agent_x_pos] = []

        for i in range(len(input_ti.observation)):
            x_obs_tmp = x_obs[sample_idx, :, :, input_ti.observation[i]].squeeze()
            pred_obs_tmp_perc = pred_obs[sample_idx, :, :, pred_ti.observation[i]]
            tmp_certainty = torch.gather(pred_obs_tmp_perc, 2, x_obs_tmp[..., None].to(torch.long)).squeeze()
            tmp_certainty = tmp_certainty.mean(dim=[0, 1])

            data[OBSERVATION_VARIABLES[i]][agent_x_pos].append(tmp_certainty.item())

    return data


def create_overlay(data: np.ndarray, grid_size: tuple[int, int], tile_size: int, border_px: int = 2):
    img_size = (grid_size[0] * tile_size, grid_size[1] * tile_size)
    
    num_bars = data.shape[2]
    bar_width = (tile_size - 2 * border_px) / num_bars
    
    overlay = Image.new('RGBA', img_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for idx in range(num_bars):
                bar_height = data[i, j, idx] * (tile_size - 2 * border_px)
                bar_region = (
                    i * tile_size + border_px + idx * bar_width,
                    j * tile_size + tile_size - border_px - bar_height,
                    i * tile_size + border_px + (idx + 1) * bar_width,
                    j * tile_size + tile_size - border_px
                )
                draw.rectangle(bar_region, fill=(255, 0, 0, 128))

    return overlay

def data_py_to_np(data: dict[str, dict[tuple[int, int], list[float]]], grid_size: tuple[int, int]) -> np.ndarray:
    data_np = {}
    for i, key in enumerate(data):
        if key not in data_np:
                data_np[key] = np.zeros((grid_size[0], grid_size[1], 2))

        for pos, values in data[key].items():
            data_np[key][pos] = np.array([np.mean(values), np.min(values)])
    return data_np