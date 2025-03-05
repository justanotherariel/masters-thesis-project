from dataclasses import dataclass
from typing import Optional

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.scoring.minigrid_heatmap import GridRenderer, RenderConfig
from src.modules.scoring.pdf_file_writer import PDFFileWriter
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

logger = Logger()


@dataclass
class MinigridEnvironmentsPDF(TransformationBlock):
    """Generate a PDF containing all grid environments grouped by dataset."""

    filename: str = "grid_environments.pdf"
    envs_per_row: int = 5
    include_indices: bool = True
    page_title: Optional[str] = None

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self.info = info
        self.renderer = GridRenderer(RenderConfig())
        return info

    def custom_transform(self, data: PipelineData, **kwargs) -> PipelineData:
        logger.info("Generating PDF with grid environments...")

        output_dir = self.info.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create PDF writer
        with PDFFileWriter(output_dir, self.filename) as pdf_writer:
            # Add main title if provided
            if self.page_title:
                pdf_writer.c.setFont("Helvetica-Bold", 20)
                pdf_writer.c.drawString(pdf_writer.margin, pdf_writer.current_y, self.page_title)
                pdf_writer.current_y -= 40  # Space after main title

            # Process each dataset group
            for dataset_group in data.grids:
                if dataset_group == DatasetGroup.ALL:
                    continue

                # Render all grids in this dataset group
                grid_images = []
                for grid in data.grids[dataset_group]:
                    grid_img = self.renderer.render_grid(grid)
                    grid_images.append(grid_img)

                # Add this dataset group to the PDF
                group_title = f"{dataset_group.name} Dataset ({len(grid_images)} environments)"
                pdf_writer.add_grid_environments(grid_images, group_title, self.envs_per_row)

        logger.info(f"PDF generated: {output_dir / self.filename}")
        return data
