"""File containing functions related to setting up the pipeline."""

from enum import Enum
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.framework.logging import Logger
from src.framework.pipeline import ModelPipeline
from src.typing.pipeline_objects import PipelineInfo

logger = Logger()


def setup_pipeline(cfg: DictConfig, output_dir: Path) -> tuple[ModelPipeline, PipelineInfo]:
    """Instantiate the pipeline.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param is_train: Whether the pipeline is used for training
    """
    logger.info("Instantiating the pipeline")

    model_cfg = cfg.model
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    pipeline_cfg = OmegaConf.create(model_cfg_dict)

    # Instantiate
    model_pipeline = instantiate(pipeline_cfg)

    # Run Setup
    info = PipelineInfo(
        debug=cfg.debug,
        output_dir=output_dir,
        trial_idx=cfg.trial_idx,
    )
    info = model_pipeline.setup(info)

    return model_pipeline, info
