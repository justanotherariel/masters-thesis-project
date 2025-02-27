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

    def add_tensor(self, tensor: torch.Tensor):
        """
        Add a tensor to the PDF as a grid of circles where the size of each circle
        represents the value (between 0 and 1).
        
        Args:
            tensor: A 2D tensor with values between 0 and 1
        """
        # # Add a page break to ensure we have a full page for the tensor
        # self.add_page_break()
        
        # Ensure tensor is 2D
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
            
        # Get tensor dimensions
        rows, cols = tensor.shape
        
        # Calculate subdivision size (square root of dimensions)
        subdivision_size_rows = int(math.sqrt(rows))
        subdivision_size_cols = int(math.sqrt(cols))
        
        # Calculate maximum circle size and grid cell size based on page dimensions
        usable_width = self.page_width - (2 * self.margin)
        usable_height = self.page_height - (2 * self.margin)
        
        # Determine grid cell size (smaller of width or height constraint)
        cell_size_width = usable_width / cols
        cell_size_height = usable_height / rows
        cell_size = min(cell_size_width, cell_size_height)
        
        # Maximum circle radius is half the cell size with some padding
        max_radius = cell_size * 0.4
        
        # Calculate grid origin (top-left corner)
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        grid_x = self.margin + (usable_width - grid_width) / 2
        grid_y = self.page_height - self.margin - (usable_height - grid_height) / 2
        
        # Draw the grid
        self.c.setLineWidth(0.5)
        self.c.setStrokeColor("lightgrey")
                
        # Draw horizontal grid lines
        for i in range(rows + 1):
            # Use thicker line for subdivisions
            if i % subdivision_size_rows == 0:
                self.c.setLineWidth(1.5)
                self.c.setStrokeColor("grey")
            else:
                self.c.setLineWidth(0.5)
                self.c.setStrokeColor("lightgrey")
                
            y = grid_y - i * cell_size
            self.c.line(grid_x, y, grid_x + grid_width, y)
        
        # Draw vertical grid lines
        for j in range(cols + 1):
            # Use thicker line for subdivisions
            if j % subdivision_size_cols == 0:
                self.c.setLineWidth(1.5)
                self.c.setStrokeColor("grey")
            else:
                self.c.setLineWidth(0.5)
                self.c.setStrokeColor("lightgrey")
                
            x = grid_x + j * cell_size
            self.c.line(x, grid_y, x, grid_y - grid_height)
        
        # Draw circles for each cell
        self.c.setStrokeColor("black")
        self.c.setFillColor("black")
        
        for i in range(rows):
            for j in range(cols):
                # Get value and calculate circle radius
                value = tensor[i, j].item()
                radius = value * max_radius
                
                # Calculate center of the cell
                center_x = grid_x + (j + 0.5) * cell_size
                center_y = grid_y - (i + 0.5) * cell_size
                
                # Draw the circle if the value is not too small
                if radius > 0.2:  # Minimum size threshold for visibility
                    self.c.circle(center_x, center_y, radius, fill=1)
                
        # Reset current y position
        self.current_y = self.margin

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