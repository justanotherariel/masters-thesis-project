import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any
import sys
from types import ModuleType
import zipfile
import time
import fcntl
import contextlib

import torch
from torch import nn

import wandb
from src.framework.logging import Logger

logger = Logger()


@contextlib.contextmanager
def file_lock(file_path, mode='w', timeout=10, check_interval=0.1):
    """Context manager for file locking.
    
    Args:
        file_path (str or Path): Path to the file to lock
        mode (str): File open mode ('r', 'w', etc.)
        timeout (float): Maximum time to wait for lock acquisition in seconds
        check_interval (float): Time between lock acquisition attempts in seconds
        
    Yields:
        file: The opened file object with an exclusive lock
        
    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout period
    """
    file_path = Path(file_path)
    lock_path = file_path.with_suffix(file_path.suffix + '.lock')
    
    # Ensure the directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Try to acquire the lock file
    while True:
        try:
            lock_file = open(lock_path, 'w+')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError:
            # Another process has the lock
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds")
            time.sleep(check_interval)
    
    try:
        # Open the actual file after acquiring the lock
        with open(file_path, mode) as f:
            yield f
    finally:
        # Release the lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        try:
            os.unlink(lock_path)  # Remove the lock file
        except OSError:
            pass  # Lock file may have been removed by another process


class ModelStorage:
    save_dir: Path
    model_hash: str
    save_to_wandb: bool

    def __init__(self, save_dir: Path, model_hash=str, saved_to_wandb: bool = True):
        self.save_dir = save_dir
        self.model_hash = model_hash
        self.save_to_wandb = saved_to_wandb
        self.db = ModelStorageDB(model_hash, save_dir, save_dir / "model_db.csv")

    def _get_model_path(self) -> Path:
        """Get the model path.

        :return: The model path.
        """
        return Path(self.save_dir) / f"{self.model_hash}.pt"

    def get_model_path(self) -> Path | None:
        """Get the model path."""
        path = self._get_model_path()

        # First, check if the model is cached locally
        if path.exists():
            # If the current run will not be saved to wandb, we can always use the cached model
            if not wandb.run:
                return path

            # If we do want to log the current run to wandb, we need to check if the model was saved
            # to the local db with a valid name
            if (name := self.db.get_name()) and name != "":
                return path

        # If we reach this point, the cached model (if there is one) was not saved to wandb
        # We thus want to delete the cached model (if there is one) and (re)train, so the the training process
        # is logged to wandb
        path.unlink(missing_ok=True)  # DB will be gc'd later
        return None

    def _get_model_checkpoint_path(self, epoch: int) -> Path:
        """Get the checkpoint path.

        :param epoch: The epoch number.
        :return: The checkpoint path.
        """
        return Path(self.save_dir) / f"{self.model_hash}_checkpoint_{epoch}.pt"

    def get_model_checkpoint_path(self, epoch: int) -> Path | None:
        """Get the checkpoint path.

        :param epoch: The epoch number.
        :return: The checkpoint path.
        """
        return self._get_model_checkpoint_path(epoch) if self._get_model_checkpoint_path(epoch).exists() else None

    def get_last_checkpoint_epoch(self) -> int:
        saved_checkpoints = list(Path(self.save_dir).glob(f"{self.model_hash}_checkpoint_*.pt"))
        if len(saved_checkpoints) > 0:
            latest_checkpoint = max([int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints])
            return latest_checkpoint
        return 0

    def save_model(
        self,
        model: nn.Module,
        location: Path | None = None,
    ) -> None:
        """Save the model in the model_directory folder."""
        location = location if location is not None else self._get_model_path()
        location.parent.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving model to {location}")
        torch.save(model, location)

        if self.save_to_wandb and wandb.run:
            self.db.add(self.model_hash, wandb.run.name)

            logger.info("Saving model to wandb")
            model_artifact = wandb.Artifact("model_trained", type="model")
            model_artifact.add_file(f"{self.save_dir}/{self.model_hash}.pt")
            wandb.log_artifact(model_artifact)

    def save_model_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
    ) -> None:
        """Save the model checkpoint in the model_directory folder."""
        location = self._get_model_checkpoint_path(epoch)
        location.parent.mkdir(exist_ok=True, parents=True)

        torch.save(model, location)

    def get_model(self, location: Path | None = None) -> Any:
        """Load the model from the model_directory folder."""
        model_path = location if location is not None else self.get_model_path()

        # Check if the model exists
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model not found in {model_path}")

        # Load model
        logger.info(f"Loading model from {model_path}")
        try:
            return torch.load(model_path, weights_only=False)
        except AttributeError as e:
            # Handle changed class names
            logger.info(f"Failed to load model, replacing class names.")
            replace_attention_class(model_path)
            return torch.load(model_path, weights_only=False)

    def get_model_checkpoint(self, epoch: int) -> Any:
        location = self.get_model_checkpoint_path(epoch)
        return self.get_model(location)


class ModelStorageDB:
    current_hash: str
    model_dir: Path
    db_path: Path

    def __init__(self, current_hash: str, model_dir: Path, db_path: Path):
        self.current_hash = current_hash
        self.model_dir = model_dir
        self.db_path = db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.touch(exist_ok=True)

    def add(self, model_hash: str, run_name: str):
        self.gc()
        try:
            with file_lock(self.db_path, mode='a') as db:
                db.write(f"{model_hash},{run_name}\n")
        except TimeoutError:
            logger.warning(f"Could not acquire lock for {self.db_path} when adding entry")

    def get_name(self, model_hash: str | None = None) -> str:
        if model_hash is None:
            model_hash = self.current_hash
        try:
            with file_lock(self.db_path, mode='r') as db:
                for line in db:
                    model, name = line.strip().split(",")
                    if model == model_hash:
                        return name
        except TimeoutError:
            logger.warning(f"Could not acquire lock for {self.db_path} when getting name, trying without lock")
            # Fallback - try without locking
            try:
                with open(self.db_path, 'r') as db:
                    for line in db:
                        model, name = line.strip().split(",")
                        if model == model_hash:
                            return name
            except Exception as e:
                logger.error(f"Error reading DB file: {e}")
        return None

    def gc(self):
        try:
            # Keep a single lock throughout the entire garbage collection process
            with file_lock(self.db_path, mode='r+') as db:
                # Read all lines from the database file
                db.seek(0)
                lines = db.readlines()
                
                # Get the list of model files that actually exist
                models = [model.stem for model in self.model_dir.glob("*.pt")]
                
                # Filter lines to keep
                lines_to_keep = []
                for line in lines:
                    if not line.strip():
                        continue
                    model_hash, _ = line.strip().split(",")
                    if model_hash in models:
                        lines_to_keep.append(line)
                
                # Truncate the file and write back the filtered lines
                db.seek(0)
                db.truncate()
                db.writelines(lines_to_keep)
                
        except TimeoutError:
            logger.warning(f"Could not acquire lock for DB garbage collection, skipping")
        except Exception as e:
            logger.error(f"Error during DB garbage collection: {e}")


def replace_attention_class(pt_file_path):
    """
    Replaces 'any_top_level_module.MultiHeadedAttention' with 'any_top_level_module.MultiHeadAttention'
    in a PyTorch pickle file (.pt).
    
    Args:
        pt_file_path (str): Path to the PyTorch pickle file
    
    Returns:
        None
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract the .pt file (which is a zip archive)
        with zipfile.ZipFile(pt_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Check if data.pkl exists in the extracted files
        archive_name = os.path.basename(pt_file_path).split('.')[0]
        data_pkl_path = os.path.join(temp_dir, archive_name, 'data.pkl')
        if not os.path.exists(data_pkl_path):
            return  # data.pkl doesn't exist, so return
        
        # Read the binary data
        with open(data_pkl_path, 'rb') as f:
            data = f.read()
        
        # Check if 'MultiHeadedAttention' exists in the data
        if b'MultiHeadedAttention' not in data:
            return  # The class doesn't exist, so return
        
        # Replace 'MultiHeadedAttention' with 'MultiHeadAttention'
        modified_data = data.replace(b'MultiHeadedAttention', b'MultiHeadAttention')
        
        # Save the modified data
        with open(data_pkl_path, 'wb') as f:
            f.write(modified_data)
        
        # Create a new zip archive with the modified content
        with zipfile.ZipFile(pt_file_path, 'w') as new_zip:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_zip.write(file_path, arcname)
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)