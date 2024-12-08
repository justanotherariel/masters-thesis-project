from pathlib import Path
from typing import Any

import torch
from torch import nn

import wandb
from src.framework.logging import Logger

logger = Logger()


class ModelStorage:
    save_dir: Path
    model_hash: str
    save_to_wandb: bool

    def __init__(self, save_dir: Path, model_hash=str, saved_to_wandb: bool = True):
        self.save_dir = save_dir
        self.model_hash = model_hash
        self.save_to_wandb = saved_to_wandb

    def _get_model_path(self) -> Path:
        """Get the model path.

        :return: The model path.
        """
        return Path(self.save_dir) / f"{self.model_hash}.pt"

    def get_model_path(self) -> Path | None:
        """Get the model path.

        :return: The model path.
        """
        return self._get_model_path() if self._get_model_path().exists() else None

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

        logger.info(f"Saving model checkpoint to {location}")
        torch.save(model, location)

    def get_model(self, location: Path | None = None) -> Any:
        """Load the model from the model_directory folder."""
        model_path = location if location is not None else self.get_model_path()

        # Check if the model exists
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model not found in {model_path}")

        # Load model
        logger.info(f"Loading model from {model_path}")
        return torch.load(model_path, weights_only=False)

    def get_model_checkpoint(self, epoch: int) -> Any:
        location = self.get_model_checkpoint_path(epoch)
        return self.get_model(location)
