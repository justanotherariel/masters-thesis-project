"""File containing functions related to setting up runtime arguments for pipelines."""
from typing import Any

from src.framework.pipeline import ModelPipeline


def setup_transform_args(
    pipeline: ModelPipeline,
    cache_args: dict[str, Any],
    fold: int = -1,
    *,
    save_model: bool = False,
    save_model_preds: bool = False,
) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param cache_args: Caching arguments
    :param train_indices: Train indices
    :param test_indices: Test indices
    :param fold: Fold number if it exists
    :param save_model: Whether to save the model to File
    :param save_model_preds: Whether to save the model predictions
    :return: Dictionary containing arguments
    """
    env_sys = {
        "cache_args": cache_args,
    }

    main_trainer = {
        "save_model": save_model,
    }

    if fold > -1:
        main_trainer["fold"] = fold

    train_sys = {
        "MainTrainer": main_trainer,
    }

    if save_model_preds:
        train_sys["cache_args"] = cache_args

    pred_sys: dict[str, Any] = {}

    train_args = {
        "env_sys": env_sys,
        "train_sys": train_sys,
        "pred_sys": pred_sys,
    }
        
    return train_args


def setup_pred_args(pipeline: ModelPipeline) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :return: Dictionary containing arguments
    """
    # pred_args = {
    #     "train_sys": {
    #         "MainTrainer": {
    #             # "batch_size": 16,
    #             # "model_folds": cfg.model_folds,
    #         },
    #     },
    # }
    pred_args: dict[str, Any] = {}

    return pred_args
