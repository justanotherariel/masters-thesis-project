"""File containing functions related to setting up runtime arguments for pipelines."""

from typing import Any

from src.framework.pipeline import ModelPipeline


def setup_transform_args(
    pipeline: ModelPipeline,
    cache_args: dict[str, Any],
    fold: int = -1,
    *,
    save_model: bool = False,
) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param cache_args: Caching arguments
    :param train_indices: Train indices
    :param test_indices: Test indices
    :param fold: Fold number if it exists
    :param save_model: Whether to save the model to File
    :return: Dictionary containing arguments
    """

    # Environment system
    env_sys = {
        "cache_args": cache_args,
    }

    # Training system
    torch_trainer = {
        "save_model": save_model,
    }

    if fold > -1:
        torch_trainer["fold"] = fold

    train_sys = {
        "TorchTrainer": torch_trainer,
    }

    # Prediction system / Scoring
    pred_sys: dict[str, Any] = {}

    # Result
    train_args = {
        "env_sys": env_sys,
        "train_sys": train_sys,
        "pred_sys": pred_sys,
    }
    return train_args
