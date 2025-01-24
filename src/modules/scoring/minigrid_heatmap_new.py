from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
import wandb
from minigrid.core.grid import Grid
import sys

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.scoring.data_transform import dataset_to_list
from src.typing.pipeline_objects import PipelineData, PipelineInfo
from src.typing.minigrid_objects import GridPosition, GridSize, VariableType
from src.modules.training.datasets.two_d_dataset import TwoDDataset
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import DatasetGroup


class MetricType(Enum):
    AGENT_POV_CERTAINTY = auto()
    AGENT_POV_ACCURACY_ALL = auto()
    AGENT_POV_ACCURACY_AVG = auto()
    
    FIELD_POV_ACCURACY = auto()
    

class MetricCalculator(Protocol):
    def calculate(self, grid: Grid, raw_data: List[List[torch.Tensor]], raw_ti: TokenIndex, preds: List[torch.Tensor], model_ti: TokenIndex) -> Dict[MetricType, np.ndarray]:
        ...
    
    def get_agent_pos(self, observation: torch.Tensor, ti: TokenIndex) -> GridPosition:
        '''Get the position of the agent in the observation.'''
        agent_pos = torch.nonzero(observation[:, :, ti.observation[3]]).squeeze()
        return GridPosition(x=agent_pos[0].item(), y=agent_pos[1].item())

class AgentPovCertaintyCalc(MetricCalculator):
    def calculate(self, grid: Grid, raw_data: List[List[torch.Tensor]], raw_ti: TokenIndex, preds: List[torch.Tensor], model_ti: TokenIndex) -> Dict[MetricType, np.ndarray]:
        grid_size = GridSize(width=grid.width, height=grid.height)
        x_obs = raw_data[0][0]
        y_obs = raw_data[1][0]
        pred_obs = preds[0]
        
        certainty = torch.zeros(x_obs.shape[0], grid_size.width, grid_size.height, len(raw_ti.observation))
        for obs_idx in raw_ti.observation:
            y_obs_tmp =  y_obs[..., raw_ti.observation[obs_idx]].squeeze()
            pred_obs_tmp = pred_obs[..., model_ti.observation[obs_idx]]
            certainty[..., obs_idx] = torch.gather(pred_obs_tmp, 3, y_obs_tmp[..., None].to(torch.long))
            
        certainty_avg = certainty.mean(dim=[1,2,3])
        metric_data_raw: Dict[tuple[int, int], List[float]] = {}
        for sample_idx in range(certainty_avg.shape[0]):
            agent_pos = self.get_agent_pos(x_obs[sample_idx], raw_ti)
            if agent_pos not in metric_data_raw:
                metric_data_raw[agent_pos] = []
            metric_data_raw[agent_pos].append(certainty_avg[sample_idx].item())
        
        metric_data = np.zeros((grid_size.width, grid_size.height, 2))
        for pos, values in metric_data_raw.items():
            metric_data[pos.x, pos.y, 0] = np.mean(values)
            metric_data[pos.x, pos.y, 1] = np.min(values)

        return {MetricType.AGENT_POV_CERTAINTY: metric_data}
    
class AgentPovAccuracyCalc(MetricCalculator):
    def calculate(self, grid: Grid, raw_data: List[List[torch.Tensor]], raw_ti: TokenIndex, preds: List[torch.Tensor], model_ti: TokenIndex) -> Dict[MetricType, np.ndarray]:
        grid_size = GridSize(width=grid.width, height=grid.height)
        x_obs = raw_data[0][0]
        y_obs = raw_data[1][0]
        pred_obs = preds[0]
        
        accuracy = torch.zeros(x_obs.shape[0], grid_size.width, grid_size.height, len(raw_ti.observation))
        for obs_idx in raw_ti.observation:
            y_obs_tmp =  y_obs[..., raw_ti.observation[obs_idx]].squeeze()
            pred_obs_tmp = torch.argmax(pred_obs[..., model_ti.observation[obs_idx]], dim=3)
            accuracy[..., obs_idx] = (pred_obs_tmp == y_obs_tmp)[..., None].float()
            
        accuracy_all = accuracy.prod(dim=3).prod(dim=2).prod(dim=1)
        accuracy_avg = accuracy.prod(dim=3).mean(dim=[1,2])
        
        metric_data_raw: Dict[tuple[int, int], Tuple[List[float], List[float]]] = {}
        for sample_idx in range(accuracy_all.shape[0]):
            agent_pos = self.get_agent_pos(x_obs[sample_idx], raw_ti)
            if agent_pos not in metric_data_raw:
                metric_data_raw[agent_pos] = ([], [])
            metric_data_raw[agent_pos][0].append(accuracy_all[sample_idx].item())
            metric_data_raw[agent_pos][1].append(accuracy_avg[sample_idx].item())
        
        metric_data = [
            np.zeros((grid_size.width, grid_size.height, 2)),
            np.zeros((grid_size.width, grid_size.height, 2))
        ]
        for pos, values in metric_data_raw.items():
            metric_data[0][pos.x, pos.y, 0] = np.mean(values[0])
            metric_data[0][pos.x, pos.y, 1] = np.min(values[0])
            
            metric_data[1][pos.x, pos.y, 0] = np.mean(values[1])
            metric_data[1][pos.x, pos.y, 1] = np.min(values[1])

        return {
            MetricType.AGENT_POV_ACCURACY_ALL: metric_data[0],
            MetricType.AGENT_POV_ACCURACY_AVG: metric_data[1]
        }

class FieldPovAccuracyCalc(MetricCalculator):
    def calculate(self, grid: Grid, raw_data: List[List[torch.Tensor]], raw_ti: TokenIndex, preds: List[torch.Tensor], model_ti: TokenIndex) -> Dict[MetricType, np.ndarray]:
        grid_size = GridSize(width=grid.width, height=grid.height)
        x_obs = raw_data[0][0]
        y_obs = raw_data[1][0]
        pred_obs = preds[0]
        
        accuracy = torch.zeros(x_obs.shape[0], grid_size.width, grid_size.height, len(raw_ti.observation))
        for obs_idx in raw_ti.observation:
            y_obs_tmp =  y_obs[..., raw_ti.observation[obs_idx]].squeeze()
            pred_obs_tmp = torch.argmax(pred_obs[..., model_ti.observation[obs_idx]], dim=3)
            accuracy[..., obs_idx] = (pred_obs_tmp == y_obs_tmp)[..., None].float()
                            
        metric_data = np.zeros((grid_size.width, grid_size.height, 2))
        metric_data[..., 0] = accuracy.prod(dim=3).mean(dim=0)
        metric_data[..., 1] = accuracy.prod(dim=3).std(dim=0)
        
        return { MetricType.FIELD_POV_ACCURACY: metric_data }

@dataclass
class RenderConfig:
    tile_size: int = 32
    border_px: int = 2

class GridRenderer:
    grid_img: Image.Image
    grid_size: GridSize
    
    def __init__(self, config: RenderConfig):
        self.config = config

    def render_grid(self, grid: Grid) -> Image.Image:
        '''Render the given grid as an image.'''
        
        img_np = grid.render(
            tile_size=self.config.tile_size,
            agent_pos=(1, 1),
            agent_dir=None,
            highlight_mask=None,
        ).astype(np.uint8)
        
        self.grid_size = GridSize(width=grid.width, height=grid.height)
        self.grid_img = Image.fromarray(img_np)
        return self.grid_img

    def create_heatmap_overlay(self, data: np.ndarray, color: tuple[float, ...]) -> tuple[Image.Image, Image.Image]:
        '''Create a heatmap overlay for the given data. Returns the raw overlay and the overlay applied to the grid image.'''
        
        img_size = self.grid_size.to_pixel_size(self.config.tile_size)
        num_bars = data.shape[2]
        bar_width = (self.config.tile_size - 2 * self.config.border_px) / num_bars
        
        overlay = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i in range(self.grid_size.width):
            for j in range(self.grid_size.height):
                for idx in range(num_bars):
                    bar_height = data[i, j, idx] * (self.config.tile_size - 2 * self.config.border_px)
                    bar_region = (
                        i * self.config.tile_size + self.config.border_px + idx * bar_width,
                        j * self.config.tile_size + self.config.tile_size - self.config.border_px - bar_height,
                        i * self.config.tile_size + self.config.border_px + (idx + 1) * bar_width,
                        j * self.config.tile_size + self.config.tile_size - self.config.border_px
                    )
                    draw.rectangle(bar_region, fill=color)
        
        applied = Image.alpha_composite(self.grid_img.convert("RGBA"), overlay)
        return overlay, applied

@dataclass
class MinigridHeatmap(TransformationBlock):
    """Generate heatmaps visualizing model predictions."""

    metric_calculators: List[str] = field(default_factory=lambda: [])

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self.info = info
        self.renderer = GridRenderer(RenderConfig())
        
        self._metric_calculators: List[MetricCalculator] = []
        for metric in self.metric_calculators:
            try:
                cls = getattr(sys.modules[__name__], metric)
                self._metric_calculators.append(cls())
            except AttributeError:
                raise ValueError(f"Metric {metric} not found.")
        return info

    def custom_transform(self, data: PipelineData, **kwargs) -> PipelineData:
        output_dir = self.info.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_group in data.grids.keys():
            if dataset_group == DatasetGroup.ALL:
                continue
            
            raw_data = dataset_to_list(data, dataset_group)
            preds = data.predictions[dataset_group]
            
            for grid_idx, grid in enumerate(data.grids[dataset_group]):
                grid_name = f"{dataset_group.name.lower()}_{grid_idx}"
                grid_img = self.renderer.render_grid(grid)
                grid_img.save(output_dir / f"grid_{grid_name}.png")
                
                grid_metrics = self._calc_grid_metrics(grid, raw_data, preds)
                self._log_grid_metrics(grid, grid_img, grid_name, grid_metrics)

        return data

    def _calc_grid_metrics(self, grid: Grid, raw_data: List[torch.Tensor], preds: List[torch.Tensor]) -> Dict[MetricType, np.ndarray]:
        grid_metrics: Dict[MetricType, np.ndarray] = {}
        
        for metric_calculator in self._metric_calculators:
            raw_ti = TwoDDataset.create_ti(self.info)
            model_ti = self.info.model_ti
            
            metric = metric_calculator.calculate(grid, raw_data, raw_ti, preds, model_ti)
            grid_metrics.update(metric)
        
        return grid_metrics

    def _log_grid_metrics(self, grid: Grid, grid_img: Image.Image,
                                grid_name: str, grid_metrics: Dict[MetricType, np.ndarray]) -> None:
        output_dir = self.info.output_dir
        colors = {
            MetricType.AGENT_POV_ACCURACY_ALL:  (255,   0,   0, 255),
            MetricType.AGENT_POV_ACCURACY_AVG:  (255,   0,   0, 255),
            MetricType.AGENT_POV_CERTAINTY:     (255,   0,   0, 255),
            MetricType.FIELD_POV_ACCURACY:      (255,   0,   0, 255)
        }
        
        wandb_masks: Dict[str, np.ndarray] = {}
        for metric_type, metric_data in grid_metrics.items():
            overlay, applied = self.renderer.create_heatmap_overlay(metric_data, colors[metric_type])

            applied.save(
                output_dir / f"grid_{grid_name}_{metric_type.name.lower()}.png"
            )
            
            np_overlay = np.array(overlay)
            wandb_masks[metric_type.name.lower()] = {
                "mask_data": np.all(np_overlay == colors[metric_type], axis=-1),
            }
                    
        Logger().log_to_external({
            f"Grid '{grid_name}'": 
            wandb.Image(grid_img, masks=wandb_masks)
        })