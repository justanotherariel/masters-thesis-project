"""File containing functions related to setting up Weights and Biases."""

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

import wandb
from src.framework.logging import Logger

logger = Logger()


def setup_wandb(
    cfg: DictConfig,
    job_type: str,
    output_dir: Path,
    name: str | None = None,
    group: str | None = None,
) -> wandb.sdk.wandb_run.Run | wandb.sdk.lib.RunDisabled | None:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param job_type: The type of job, e.g. Training, CV, etc.
    :param output_dir: The directory to the Hydra outputs.
    :param name: The name of the run.
    :param group: The namer of the group of the run.
    """
    logger.debug("Initializing Weights & Biases")

    config = OmegaConf.to_container(cfg, resolve=True)

    # Get the model name
    model_target = get_nested_value(config, "model.train_sys.steps.0.model._target_")
    if model_target:
        model_name = model_target.split(".")[-1]
        config["model"]["train_sys"]["steps"][0]["model"]["name"] = model_name

    run = wandb.init(
        config=replace_list_with_dict(config),  # type: ignore[arg-type]
        project="Thesis",
        entity="a-ebersberger-tu-delft",
        name=name,
        group=group,
        job_type=job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=str(output_dir),
        reinit=True,
    )

    if (
        isinstance(run, wandb.sdk.lib.RunDisabled) or run is None
    ):  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
        raise RuntimeError("Failed to initialize Weights & Biases")

    if cfg.wandb.log_config:
        logger.debug("Uploading config files to Weights & Biases")

        # Get the config file name
        curr_config = "conf/train.yaml"

        # Get the model file name
        if isinstance(OmegaConf.load(curr_config).defaults[2], str):
            model_name = OmegaConf.load(curr_config).defaults[2].split("@")[0]
            model_path = f"conf/{model_name}.yaml"
        else:
            model_name = OmegaConf.load(curr_config).defaults[2].model
            model_path = f"conf/model/{model_name}.yaml"

        # Store the config as an artefact of W&B
        artifact = wandb.Artifact("train_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"
        artifact.add_file(str(config_path), "config.yaml")
        artifact.add_file(curr_config)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    if cfg.wandb.log_code.enabled:
        logger.debug("Uploading code files to Weights & Biases")

        run.log_code(
            root=".",
            exclude_fn=cast(
                Callable[[str, str], bool],
                lambda abs_path, root: re.match(
                    cfg.wandb.log_code.exclude,
                    Path(abs_path).relative_to(root).as_posix(),
                )
                is not None,
            ),
        )

    logger.info("Done initializing Weights & Biases")
    return run


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, _v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def replace_list_with_dict(o: object) -> object:
    """Recursively replace lists with integer index dicts.

    This is necessary for wandb to properly show any parameters in the config that are contained in a list.

    :param o: Initially the dict, or any object recursively inside it.
    :return: Integer index dict.
    """
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = replace_list_with_dict(v)
    elif isinstance(o, list):
        o = {i: replace_list_with_dict(v) for i, v in enumerate(o)}
    return o


def get_nested_value(dictionary, path, default=None):
    """
    Safely access nested dictionary values using a list of keys or dot notation string.
    Returns default value if path doesn't exist.

    Args:
        dictionary (dict): The dictionary to search in
        path (str|list): Path to value, either as a dot-separated string or list of keys
        default: Value to return if path doesn't exist (default: None)

    Examples:
        get_nested_value(config, 'model.train_sys.steps.0.model._target_')
        get_nested_value(config, ['model', 'train_sys', 'steps', 0, 'model', '_target_'])
    """
    keys = path.split(".") if isinstance(path, str) else path

    current = dictionary
    for key in keys:
        try:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            if isinstance(current, (dict, list)):
                current = current[key] if isinstance(current, dict) else current[int(key)]
            else:
                return default
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    return current
