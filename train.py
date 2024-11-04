"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import wandb
import coloredlogs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.train_config import TrainConfig
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_transform_args
from src.setup.setup_wandb import setup_wandb
from src.utils.lock import Lock
from src.utils.set_torch_seed import set_torch_seed
from src.framework.logging import Logger

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the model pipeline. Entry point for Hydra which loads the config file."""
    
    # Install coloredlogs
    coloredlogs.install()
        
    # Run the train config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_train(cfg)


def run_train(cfg: DictConfig) -> None:
    """Run the model pipeline."""
    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Setup the logger
    logger = Logger(output_dir)
    logger.log_section_separator("Thesis: Sparse Transformer")
    
    # Set seed
    set_torch_seed()

    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline
    logger.info("Setting up the pipeline")
    model_pipeline = setup_pipeline(cfg)
    _ = model_pipeline.setup()

    # Cache arguments for x_sys
    cache_data_path = Path(cfg.cache_path)
    cache_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{cache_data_path}",
    }

    logger.log_section_separator("Train model pipeline")
    transform_args = setup_transform_args(model_pipeline, cache_args)
    _ = model_pipeline.transform(**transform_args)

    # if y is None:
    #     y = y_new

    # if len(test_indices) > 0:
    #     logger.log_section_separator("Scoring")
    #     scorer = instantiate(cfg.scorer)
    #     score = scorer(y[test_indices], predictions[test_indices])
    #     logger.info(f"Score: {score}")

    #     if wandb.run:
    #         wandb.log({"Score": score})

    wandb.finish()


if __name__ == "__main__":
    main()
