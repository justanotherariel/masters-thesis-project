import io
import math
from pathlib import Path

import torch
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


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

        # Spacing
        self.arrow_width = 30
        self.idx_width = 20  # Width for sample index display

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

    def add_row(
        self,
        idx: int,
        x_obs: Image.Image,
        x_action: str,
        y_obs: Image.Image,
        y_reward: float,
        pred_obs: Image.Image,
        pred_reward: float,
        correct_obs: bool,
        correct_reward: bool,
    ):
        """
        Add a row with observation images, action text, and rewards.

        Args:
            idx: Sample index within the dataset
            x_obs: Input observation (PIL Image)
            x_action: Action index
            y_obs: Target observation (PIL Image)
            y_reward: Target reward value
            pred_obs: Predicted observation (PIL Image)
            pred_reward: Predicted reward value
            correct_obs: Whether observation prediction was correct
            correct_reward: Whether reward prediction was correct
        """
        # Check if we need a new page
        if self.current_y - self.row_height < self.margin:
            self.c.showPage()
            self.current_y = self.page_height - self.margin

        # Calculate positions
        x_positions = []
        x_positions.append(self.margin - 20)  # idx display
        x_positions.append(x_positions[-1] + self.idx_width)  # x_obs
        x_positions.append(x_positions[-1] + self.image_width)  # action text
        x_positions.append(x_positions[-1] + self.image_width)  # arrow
        x_positions.append(x_positions[-1] + 50)  # rewards
        x_positions.append(x_positions[-1] + self.image_width)  # y_obs
        x_positions.append(x_positions[-1] + self.image_width + 20)  # separator
        x_positions.append(x_positions[-1] + 20)  # pred_obs

        # Draw the sample index vertically
        self.c.saveState()
        self.c.setFont("Helvetica-Bold", 10)
        self.c.translate(x_positions[0] + 10, self.current_y - self.image_height/2)  # Position in the middle of the row
        self.c.rotate(90)  # Rotate 90 degrees to write vertically
        idx_text = f"#{idx}"
        self.c.drawString(0, 0, idx_text)  # The 0,0 position is now rotated
        self.c.restoreState()

        # Draw images and elements
        for img, pos in [(x_obs, 1), (y_obs, 5), (pred_obs, 7)]:
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
        text_x = x_positions[2] + (self.image_width - text_width) / 2
        text_y = self.current_y - self.image_height / 2
        self.c.drawString(text_x, text_y, x_action)

        # Add arrow
        arrow_color = "green" if (correct_obs and correct_reward) else "red"
        self._add_arrow(x_positions[3], self.current_y - self.image_height / 2, arrow_color)

        # Add reward values
        self.c.setFont("Courier", 9)  # Use monospace font for aligned numbers
        self.c.setFillColor("black" if correct_reward else "red")
        reward_x = x_positions[4]
        # Target reward (top)
        target_text = f"Target: {self._format_reward(y_reward)}"
        self.c.drawString(reward_x, self.current_y - self.image_height / 3, target_text)
        # Predicted reward (bottom)
        pred_text = f"Pred  : {self._format_reward(pred_reward)}"
        self.c.drawString(reward_x, self.current_y - self.image_height * 2 / 3, pred_text)
        self.c.setFillColor("black")

        # Add vertical separator
        self._add_vertical_separator(x_positions[6], self.current_y - self.image_height, self.image_height)

        # Update current_y position
        self.current_y -= self.row_height

    def add_tensor(
        self,
        tensor: torch.Tensor,
        highlight: list[tuple[int, str | None, str | None]] = None,
    ):
        """
        Add a tensor to the PDF as a grid of circles where the size of each circle
        represents the value (between 0 and 1).

        Args:
            tensor: A 2D tensor with values between 0 and 1
            highlight: List of tuples (idx, row_color, col_color) specifying which 
                      rows/columns to highlight and with what colors. If row_color or 
                      col_color is None, that dimension won't be highlighted.
        """
        # Ensure tensor is 2D
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")

        # Initialize highlight list if None
        if highlight is None:
            highlight = []

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

        # First, draw highlights if any
        for idx, row_color, col_color in highlight:
            # Validate index is within bounds
            if idx < 0 or idx >= max(rows, cols):
                continue
                
            # Draw row highlight if row_color is provided and index is valid
            if row_color is not None and idx < rows:
                self.c.setFillColor(row_color)
                self.c.setFillAlpha(0.3)  # Semi-transparent
                self.c.rect(
                    grid_x,
                    grid_y - (idx + 1) * cell_size,
                    grid_width,
                    cell_size,
                    fill=1,
                    stroke=0
                )
                
            # Draw column highlight if col_color is provided and index is valid
            if col_color is not None and idx < cols:
                self.c.setFillColor(col_color)
                self.c.setFillAlpha(0.3)  # Semi-transparent
                self.c.rect(
                    grid_x + idx * cell_size,
                    grid_y - grid_height,
                    cell_size,
                    grid_height,
                    fill=1,
                    stroke=0
                )
                
        # Reset fill opacity
        self.c.setFillAlpha(1.0)

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
                if radius > 0.1:  # Minimum size threshold for visibility
                    self.c.circle(center_x, center_y, radius, fill=1)

        # Reset current y position
        self.current_y = self.margin

    def add_grid_environments(self, environments: list[Image.Image], title: str, envs_per_row: int = 5):
        """
        Add a section of grid environment images to the PDF with a title.

        Args:
            environments: List of environment images to display
            title: Title for this group of environments
            envs_per_row: Number of environments to display per row
        """
        # Calculate image dimensions
        env_width = (self.usable_width - ((envs_per_row - 1) * 10)) / envs_per_row  # 10px spacing between images
        env_height = env_width  # Keep square aspect ratio for grid environments

        # Check if we need a new page for title
        if self.current_y - 40 - env_height < self.margin:
            self.c.showPage()
            self.current_y = self.page_height - self.margin

        # Add title with some padding
        self.c.setFont("Helvetica-Bold", 16)
        self.c.drawString(self.margin, self.current_y, title)
        self.current_y -= 20  # Space after title

        # Process environments in chunks of envs_per_row
        for i in range(0, len(environments), envs_per_row):
            # Get the chunk of environments for this row
            row_envs = environments[i : i + envs_per_row]

            # Draw each environment in the row
            for j, env in enumerate(row_envs):
                img_buffer = io.BytesIO()
                env.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                x_pos = self.margin + j * (env_width + 10)  # 10px spacing
                self.c.drawImage(
                    ImageReader(img_buffer),
                    x_pos,
                    self.current_y - env_height,
                    width=env_width,
                    height=env_height,
                    preserveAspectRatio=True,
                )

                # Optional: Add index number below each environment
                self.c.setFont("Helvetica", 8)
                self.c.drawString(x_pos + env_width / 2 - 10, self.current_y - env_height - 10, f"#{i+j+1}")

            # Update y position for next row
            self.current_y -= env_height + 50  # Add some space between rows

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
