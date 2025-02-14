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
    model_target = cfg.model.train_sys.steps[0].model._target_
    if model_target:
        model_config_name = model_target.split(".")[-1]
        config["model"]["train_sys"]["steps"][0]["model"]["name"] = model_config_name

    run = wandb.init(
        config=replace_list_with_dict(config),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=name,
        group=group,
        job_type=cfg.wandb.job_type,
        settings=wandb.Settings(code_dir="."),
        dir=str(output_dir),
    )

    if (
        isinstance(run, wandb.sdk.lib.RunDisabled) or run is None
    ):  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
        raise RuntimeError("Failed to initialize Weights & Biases")

    if cfg.wandb.log_config:
        logger.debug("Uploading config files to Weights & Biases")

        # Get the human-readable main config file path
        main_config_path = "conf/train.yaml"

        # Get the human-readable model config file path
        if isinstance(OmegaConf.load(main_config_path).defaults[3], str):
            model_config_name = OmegaConf.load(main_config_path).defaults[3].split("@")[0]
            model_config_path = f"conf/{model_config_name}.yaml"
        else:
            model_config_name = OmegaConf.load(main_config_path).defaults[3].model
            model_config_path = f"conf/model/{model_config_name}.yaml"

        # Get the complete config file path
        complete_config_path = str(output_dir / ".hydra/config.yaml")

        # Store the config as an artefact of W&B
        artifact = wandb.Artifact("train_config", type="config")
        artifact.add_file(complete_config_path, name="complete_config.yaml")
        artifact.add_file(main_config_path)
        artifact.add_file(model_config_path)

        if cfg.wandb.sweep_param_path:
            sweep_param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH", None)
            artifact.add_file(sweep_param_path, name="sweep_param.yaml")

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
    return run


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            logger.info(f"Removing {k}={v} from environment")
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
