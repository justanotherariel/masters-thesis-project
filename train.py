"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""

# Modify sys argv to include ++ for hydra to append to the config
import sys
blacklist = ["model"]
sys.argv = sys.argv[:1] + [f"++{arg}" if arg.split('=')[0] not in blacklist else arg for arg in sys.argv[1:]]

import collections
import multiprocessing
import os
from typing import List
from pathlib import Path
import shutil

import hydra
import wandb
import coloredlogs
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import numpy as np

from src.typing.pipeline_objects import PipelineData
from src.config.train_config import TrainConfig
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_transform_args
from src.setup.setup_wandb import setup_wandb, reset_wandb_env
from src.utils.lock import Lock
from src.utils.set_torch_seed import set_torch_seed
from src.framework.logging import Logger
from src.modules.training.torch_trainer import append_to_dict, average_dict, log_dict

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)

# Define Classes for multiprocessing
Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple("WorkerInitData", ("config", "output_dir"))
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("accuracies"))

# Set up the logger
logger = Logger()

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the model pipeline. Entry point for Hydra which loads the config file."""
    
    # Install coloredlogs
    coloredlogs.install()
    
    # Check if the config is valid
    if cfg.trial_idx is not None and cfg.trial_idx >= cfg.n_trials:
        raise ValueError("Trial index smaller than number of trials")
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Print given parameters
    logger.info("Running with the following parameters:")
    for arg in sys.argv[1:]:
        if arg.startswith("++"):
            arg = arg[2:]
        logger.info(arg)
    logger.info("Output directory: %s", output_dir)
    logger.info("End of parameters")
    
    # Check if Random Seeds Initialization is enabled
    if cfg.trial_idx == None and cfg.n_trials > 1:
        run_trials(cfg, output_dir)
    else:
        run_train(cfg, output_dir)


def run_trials(cfg: DictConfig, output_dir: Path) -> None:
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers: List[Worker] = []
    for _ in range(cfg.n_trials):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_train_trial, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    # Create an 'average' run of all trials
    if cfg.wandb.enabled:
        logger.info("Initializing W&B Average Run")
        sweep_run = setup_wandb(
            cfg=cfg,
            output_dir=output_dir
        )
        cfg.wandb.run_name_base = sweep_run.name
        cfg.wandb.group_id = wandb.util.generate_id()
        cfg.wandb.sweep_id = os.environ.get("WANDB_SWEEP_ID", None)
        cfg.wandb.sweep_param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH", None)
        logger.info("Group ID: %s", cfg.wandb.group_id)
        logger.info("Done Initializing W&B Average Run\n\n")
    
    # Create the processes and execute one process after another
    accuracies = {}
    for trial_idx in range(cfg.n_trials):
        cfg.trial_idx = trial_idx
        
        # Run the trial
        worker = workers[trial_idx]
        output_dir_trial = output_dir / f"trial_{trial_idx}"
        output_dir_trial.mkdir(parents=True, exist_ok=True)
        
        # Copy .hydra directory from output_dir to output_dir_trial
        hydra_dir = output_dir / ".hydra"
        if hydra_dir.exists():
            shutil.copytree(hydra_dir, output_dir_trial / ".hydra", dirs_exist_ok=True)
        
        worker.queue.put(WorkerInitData(config=cfg, output_dir=output_dir_trial))
        
        # Get the result
        result: WorkerDoneData = sweep_q.get()
        worker.process.join()
        
        # Log the results
        for ds, acc in result.accuracies.items():
            if ds not in accuracies:
                accuracies[ds] = {}
            append_to_dict(accuracies[ds], acc)
    
    logger.info("--- Results ---")
    for ds in accuracies:
        for key in accuracies[ds]:
            logger.info(f"{ds.name.capitalize()}/{key}: {np.mean(accuracies[ds][key]):.4f} "
                        f"(± {np.std(accuracies[ds][key]):.4f})")
    logger.info("--- Results End ---")
                
    # Average the results and log to sweep run
    for ds in accuracies.keys():
        accuracies[ds] = average_dict(accuracies[ds])
        log_dict(accuracies[ds], ds.name.capitalize(), commit=True)

    sweep_run.finish()

def run_train_trial(sweep_q, worker_q):
    reset_wandb_env()
    worker_data: WorkerInitData = worker_q.get()
    accuracies = run_train(worker_data.config, worker_data.output_dir)
    sweep_q.put(WorkerDoneData(accuracies=accuracies))

def run_train(cfg: DictConfig, output_dir: Path) -> None:
    """Run the model pipeline."""
    logger.log_section_separator("Thesis: Sparse Transformer")
    
    # Set trial index if not provided
    cfg.trial_idx = cfg.trial_idx if cfg.trial_idx is not None else 0
    
    # Set seed
    cfg.seed += cfg.trial_idx
    set_torch_seed(cfg.seed)
    
    if cfg.debug:
        cfg.wandb.enabled = False

    if cfg.wandb.enabled:
        run_name = cfg.wandb.run_name_base + f"_{cfg.trial_idx}" if cfg.wandb.run_name_base else None
        
        setup_wandb(
            cfg=cfg,
            output_dir=output_dir,
            name=run_name,
            group=cfg.wandb.group_id,
        )

    # Setup the pipeline
    logger.info("Setting up the pipeline")
    model_pipeline, _info = setup_pipeline(cfg, output_dir=output_dir)

    # Cache arguments for x_sys
    cache_data_path = Path(cfg.cache_path)
    cache_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{cache_data_path}",
    }

    transform_args = setup_transform_args(model_pipeline, cache_args)
    data: PipelineData = model_pipeline.transform(**transform_args)

    wandb.finish()

    return data.accuracies

if __name__ == "__main__":
    main()
