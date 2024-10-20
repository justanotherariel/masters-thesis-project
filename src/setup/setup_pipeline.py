"""File containing functions related to setting up the pipeline."""

from enum import Enum
from typing import Any

from src.framework.pipeline import ModelPipeline
from src.framework.logging import Logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = Logger()


def setup_pipeline(cfg: DictConfig) -> ModelPipeline:
    """Instantiate the pipeline.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param is_train: Whether the pipeline is used for training
    """
    logger.info("Instantiating the pipeline")

    model_cfg = cfg.model
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    model_cfg_dict = update_model_cfg_test_size(model_cfg_dict)
    pipeline_cfg = OmegaConf.create(model_cfg_dict)

    model_pipeline = instantiate(pipeline_cfg)
    logger.debug(f"Pipeline: \n{model_pipeline}")

    return model_pipeline


def update_model_cfg_test_size(
    cfg: dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None,
    test_size: float = -1.0,
) -> dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None:
    """Update the test size in the model config.

    :param cfg: The model config.
    :param test_size: The test size.

    :return: The updated model config.
    """
    if cfg is None:
        raise ValueError("cfg should not be None")
    if isinstance(cfg, dict):
        for model in cfg["train_sys"]["steps"]:
            if model["_target_"] == "src.modules.training.main_trainer.MainTrainer":
                model["n_folds"] = test_size
    return cfg
