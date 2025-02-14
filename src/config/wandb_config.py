"""Schemas for the Weights & Biases configuration."""

from dataclasses import dataclass, field


@dataclass
class WandBLogCodeConfig:
    """Schema for the code logging to Weights & Biases.

    :param enabled: Whether to log the code to Weights & Biases.
    :param exclude: Regex of files to exclude from logging.
    """

    enabled: bool = False
    exclude: str = ""


@dataclass
class WandBConfig:
    """Schema for the Weights & Biases configuration.

    :param enabled: Whether to log to Weights & Biases.
    :param log_config: Whether to log the config to Weights & Biases.
    :param log_code: Whether to log the code to Weights & Biases.
    :param tags: Optional list of tags for the run.
    :param notes: Optional notes for the run.
    """

    enabled: bool

    project: str
    entity: str
    job_type: str

    run_name_base: str | None = None  # Runtime parameter
    group_id: str | None = None  # Runtime parameter
    sweep_id: str | None = None  # Runtime parameter
    sweep_param_path: str | None = None  # Runtime parameter

    log_config: bool = False
    log_code: WandBLogCodeConfig = field(default_factory=lambda: WandBLogCodeConfig())
