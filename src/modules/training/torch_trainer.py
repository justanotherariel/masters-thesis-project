"""TorchTrainer is a module that allows for the training of Torch models."""

import contextlib
import copy
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypeVar

import numpy as np
import torch
from annotated_types import Ge, Gt
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.utils.model_storage import ModelStorage
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .loss import BaseLoss
from .accuracy import BaseAccuracy
from .models.base import BaseModel

logger = Logger()

T = TypeVar("T", bound=Dataset)  # type: ignore[type-arg]
T_co = TypeVar("T_co", covariant=True)


def custom_collate(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch[0], batch[1]
    return X, y


@dataclass
class ModelStorageConf:
    save_model_to_disk: bool = field(default=True, repr=False, compare=False)
    save_model_to_wandb: bool = field(default=True, repr=False, compare=False)
    save_checkpoints_to_disk: bool = field(default=True, repr=False, compare=False)

    save_checkpoint_every_x_epochs: Annotated[int, Gt(0)] = field(default=0, repr=False, compare=False)
    resume_training_from_checkpoint: bool = field(default=True, repr=False, compare=False)

    save_directory: Path = field(default=Path("tm/"), repr=False, compare=False)


@dataclass
class TorchTrainer(TransformationBlock):
    """The Main Torch Trainer"""

    # Modules
    model: BaseModel
    loss: BaseLoss
    accuracy: BaseAccuracy
    optimizer: functools.partial[Optimizer]
    scheduler: Callable[[Optimizer], LRScheduler] | None = None
    model_storage_conf: ModelStorageConf = field(default_factory=ModelStorageConf, init=True, repr=False, compare=False)
    dataloader_conf: dict[str, Any] = field(default_factory=dict, repr=False)

    # Training parameters
    epochs: Annotated[int, Gt(0)] = 10
    patience: Annotated[int, Gt(0)] = -1  # Early stopping
    batch_size: Annotated[int, Gt(0)] = 32
    use_mixed_precision: bool = field(default=False)
    validate_every_x_epochs: Annotated[int, Gt(0)] = 1
    load_all_batches_to_gpu: bool = field(default=False)

    # Predction parameters
    discrete: bool = field(default=True)
    to_predict: DatasetGroup = field(default=DatasetGroup.VALIDATION, repr=False, compare=False)

    # Parameters relevant for Hashing
    n_folds: Annotated[int, Ge(1)] = field(default=1, init=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post init method for the TorchTrainer class."""

        # Setup ModelStorage
        self.model_storage_conf = ModelStorageConf(**self.model_storage_conf)
        ms = self.model_storage_conf
        if ms.save_model_to_wandb and not ms.save_model_to_disk:
            raise ValueError("Cannot save model to wandb without saving to disk.")

        # self.to_predict is a string, but we want to store it as a DataSetTypes
        self.to_predict = DatasetGroup[self.to_predict]

        # Initialize variables
        self.best_model_state_dict: dict[Any, Any] = {}

        super().__post_init__()

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """
        self._setup_info = info

        # Setup Fold
        self.current_fold = info.trial_idx

        # Setup ModelStorage (wait until self._hash is set)
        ms = self.model_storage_conf
        self.model_storage = ModelStorage(
            ms.save_directory,
            self.get_hash(),
            ms.save_model_to_wandb,
        )

        info.model_train_on_discrete = self.discrete
        info.model_ti = self.model.get_dataset_cls().create_ti(info)

        # Setup Model and Loss
        self.model.setup(info)
        self.loss.setup(info)
        self.accuracy.setup(info)

        # Move model to fastest device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using Device: {self.device}{' (debug)' if info.debug else ''}")
        self.model.module.to(self.device)

        # Disable Dataloader parallelism if debugging
        if info.debug:
            logger.info("Debug Mode: Disabling Dataloader Parallelism")
            self.dataloader_conf = {}

        # Set optimizer
        self.initialized_optimizer = self.optimizer(self.model.module.parameters())

        # Set scheduler
        self.initialized_scheduler: LRScheduler | None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)
        else:
            self.initialized_scheduler = None

        # Mixed precision
        if self.use_mixed_precision:
            logger.info("Enabling Mixed Precision Training")
            self.scaler = torch.GradScaler(device=self.device.type)
            torch.set_float32_matmul_precision("high")

        return info

    def custom_transform(self, data: PipelineData, **train_args: Any) -> PipelineData:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments.
            - train_indices: The indices to train on.
            - validation_indices: The indices to validate on.
            - save_model: Whether to save the model.
        :return: The input and output of the system.
        """

        # Train or load model from disk
        if self.model_storage.get_model_path() is not None:
            logger.info(
                f"Model exists in {self.model_storage.get_model_path()}. Loading model...",
            )
            logger.log_to_external({"Cached Model": self.model_storage.db.get_name()})
            model = self.model_storage.get_model()
            self.model.module.load_state_dict(model.state_dict())
        else:
            current_time = time.time()
            self._model_train(data)
            mins = (time.time() - current_time) // 60
            secs = (time.time() - current_time) % 60
            min_str = "minutes" if mins != 1 else "minute"
            sec_str = "seconds" if secs != 1.0 else "second"
            logger.info(f"Training took {f'{mins} {min_str} and ' if mins > 0 else ''}{secs:.2f} {sec_str}")

        # Evaluate the model
        if DatasetGroup.TRAIN in self.to_predict:
            logger.info("Running inference on the training set")
            loader = self.create_dataloader(data, DatasetGroup.TRAIN, shuffle=False)
            data.predictions.update({DatasetGroup.TRAIN: self.predict_on_loader(loader)})

        if DatasetGroup.VALIDATION in self.to_predict:
            logger.info("Running inference on the validation set")
            loader = self.create_dataloader(data, DatasetGroup.VALIDATION, shuffle=False)
            data.predictions.update({DatasetGroup.VALIDATION: self.predict_on_loader(loader)})

        return data

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> list[tuple[Tensor, ...]] | list[Tensor]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.model.module.eval()
        predictions = []

        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = moveTo(data[0], None, self.device)
                y_pred = self.model.forward(X_batch)
                y_pred = tuple(y.to("cpu") for y in y_pred) if isinstance(y_pred, tuple) else y_pred.to("cpu")
                if isinstance(y_pred, tuple):
                    if predictions == []:
                        predictions = [[] for _ in range(len(y_pred))]
                    for idx, pred in enumerate(y_pred):
                        predictions[idx].extend(pred)
                else:
                    predictions.extend(y_pred)
        return [torch.stack(pred) for pred in predictions]

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        result = f"{self._hash}_{self.n_folds}"
        if hasattr(self, 'current_fold') and self.current_fold != -1:
            result += f"_f{self.current_fold}"
        return result

    def create_dataloader(
        self,
        data: PipelineData,
        indices: DatasetGroup,
        shuffle: bool = True,
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        # Create datasets
        dataset = self.model.get_dataset_cls()(data, indices, discretize=self.discrete)
        dataset.setup(self._setup_info)

        # Check if the dataset has a custom collate function
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None

        # Create dataloaders
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **self.dataloader_conf,
        )
        return loader

    def _model_train(
        self,
        data: PipelineData,
    ):
        # Log the model being trained
        logger.info(f"Training model: {self.model.__class__.__name__}")

        # Create dataloaders
        train_loader = self.create_dataloader(data, DatasetGroup.TRAIN)
        validation_loader = self.create_dataloader(data, DatasetGroup.VALIDATION)

        # Resume from checkpoint if enabled and checkpoint exists
        start_epoch = 0
        if self.model_storage_conf.resume_training_from_checkpoint:
            last_checkpoint = self.model_storage.get_last_checkpoint_epoch()
            if last_checkpoint > 0:
                logger.info("Resuming training from checkpoint")
                model = self.model_storage.get_model_checkpoint(last_checkpoint)
                self.model.module.load_state_dict(model.state_dict())
                start_epoch = last_checkpoint + 1

        # Train the model
        logger.info(
            f"Training model for {self.epochs:,} epochs"
            f"{', starting at epoch ' + str(start_epoch) if start_epoch > 0 else ''}"
        )
        self._model_training_loop(
            train_loader,
            validation_loader,
            start_epoch,
        )
        logger.info(
            f"Done training the model: {self.model.module.__class__.__name__}",
        )

        # Revert to the best model
        if self.best_model_state_dict:
            logger.info(
                f"Reverting to model with best validation loss {self.lowest_val_loss}",
            )
            self.model.module.load_state_dict(self.best_model_state_dict)

        # Save the model
        if self.model_storage_conf.save_model_to_disk:
            self.model_storage.save_model(self.model.module)

    def _model_training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        validation_loader: DataLoader[tuple[Tensor, ...]],
        start_epoch: int = 0,
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the validation data.
        :param validation_loader: Dataloader for the training data. (can be empty)
        """

        logger.external_define_metric(f"Train/Loss", "Epoch")
        logger.external_define_metric(f"Validation/Loss", "Epoch")

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        # Track validation loss for early stopping
        self.lowest_val_loss = np.inf
        self.last_val_loss = np.inf

        for epoch in range(start_epoch, self.epochs):
            # Train using train_loader
            train_loss, accuarcy = self.train_one_epoch(train_loader, epoch)
            logger.debug(f"Epoch {epoch} Train Loss: {train_loss}")

            # Log train loss
            logger.log_to_external(
                message={
                    f"Train/Loss": train_loss,
                    "Epoch": epoch,
                },
            )
            log_dict(accuarcy, epoch, f"Train")

            # Step the scheduler
            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.model_storage_conf.save_checkpoints_to_disk:
                # Save checkpoint
                self._save_model(
                    self.get_model_checkpoint_path(epoch),
                    save_to_external=False,
                    quiet=True,
                )

                # Remove old checkpoints
                if (
                    self.checkpointing_keep_every == 0 or epoch % self.checkpointing_keep_every != 0
                ) and self.get_model_checkpoint_path(epoch - 1).exists():
                    self.get_model_checkpoint_path(epoch - 1).unlink()

            # Compute validation loss
            if len(validation_loader) > 0 and epoch % self.validate_every_x_epochs == 0:
                self.last_val_loss, accuarcy = self.val_one_epoch(
                    validation_loader,
                    epoch=epoch,
                )
                logger.debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                logger.log_to_external(
                    message={
                        f"Validation/Loss": self.last_val_loss,
                        "Epoch": epoch,
                    },
                )
                log_dict(accuarcy, epoch, f"Validation")

                # Early stopping
                if self.patience_exceeded():
                    logger.info(f"Early stopping after {self.early_stopping_counter} epochs")
                    logger.log_to_external(
                        message={f"Epochs": (epoch + 1) - (self.patience * self.validate_every_x_epochs)},
                    )
                    break

            # Log the trained epochs to wandb if we finished training
            logger.log_to_external(message={f"Epochs": epoch + 1})

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> tuple[float, dict[str, float]]:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        epoch_loss = []
        epoch_accuracy: dict[str, list[float]] = {}

        if self.load_all_batches_to_gpu and not hasattr(self, "preloaded_train_batches"):
            self.preloaded_train_batches = list(dataloader)
            batches = self.preloaded_train_batches
            for batch_idx in range(len(batches)):
                batches[batch_idx] = moveTo(batches[batch_idx][0], batches[batch_idx][1], self.device)

        if self.load_all_batches_to_gpu:
            pbar = tqdm(
                self.preloaded_train_batches,
                unit="batch",
                desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
            )
        else:
            pbar = tqdm(
                dataloader,
                unit="batch",
                desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
            )

        self.model.module.train()
        for batch in pbar:
            x_batch, y_batch = batch
            if not self.load_all_batches_to_gpu:
                x_batch, y_batch = moveTo(x_batch, y_batch, self.device)

            # Forward pass
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model.forward(x_batch)
                loss = self.loss(y_pred, y_batch)
                acc = self.accuracy(y_pred, y_batch, x_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.initialized_optimizer.step()

            # Save metrics
            epoch_loss.append(loss.item())
            append_to_dict(epoch_accuracy, acc)
            pbar.set_postfix(loss=sum(epoch_loss) / len(epoch_loss))

        return sum(epoch_loss) / len(epoch_loss), average_dict(epoch_accuracy)

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        epoch_loss = []
        epoch_accuracy: dict[str, list[float]] = {}

        if self.load_all_batches_to_gpu and not hasattr(self, "preloaded_validation_batches"):
            self.preloaded_validation_batches = list(dataloader)
            batches = self.preloaded_validation_batches
            for batch_idx in range(len(batches)):
                batches[batch_idx] = moveTo(batches[batch_idx][0], batches[batch_idx][1], self.device)

        if self.load_all_batches_to_gpu:
            pbar = tqdm(
                self.preloaded_validation_batches,
                unit="batch",
                desc=f"Epoch {epoch} Valid",
            )
        else:
            pbar = tqdm(
                dataloader,
                unit="batch",
                desc=f"Epoch {epoch} Valid",
            )

        self.model.module.eval()
        with torch.no_grad():
            for batch in pbar:
                x_batch, y_batch = batch
                if not self.load_all_batches_to_gpu:
                    x_batch, y_batch = moveTo(x_batch, y_batch, self.device)

                # Forward pass
                y_pred = self.model.forward(x_batch)
                loss = self.loss(y_pred, y_batch)
                acc = self.accuracy(y_pred, y_batch, x_batch)

                # Save metrics
                epoch_loss.append(loss.item())
                append_to_dict(epoch_accuracy, acc)
                pbar.set_postfix(loss=sum(epoch_loss) / len(epoch_loss))
        return sum(epoch_loss) / len(epoch_loss), average_dict(epoch_accuracy)

    def patience_exceeded(self) -> bool:
        """Check if early stopping should be performed.

        :return: Whether to perform early stopping.
        """
        # Store the best model so far based on validation loss
        if self.patience != -1:
            if self.last_val_loss < self.lowest_val_loss:
                self.lowest_val_loss = self.last_val_loss
                self.best_model_state_dict = copy.deepcopy(self.model.module.state_dict())
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    return True
        return False


def moveTo(
    x_batch: torch.Tensor | tuple[torch.Tensor, ...],
    y_batch: None | torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Move tensor(s) to device."""
    x_batch = tuple(x.to(device) for x in x_batch) if isinstance(x_batch, tuple) else x_batch.to(device)

    if y_batch is None:
        return x_batch

    y_batch = tuple(y.to(device) for y in y_batch) if isinstance(y_batch, tuple) else y_batch.to(device)

    return x_batch, y_batch

def append_to_dict(
    target: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Append the values of source to target."""
    for key, value in source.items():
        if key not in target:
            target[key] = []
        target[key].append(value)
    return target

def average_dict(
    target: dict[str, list[float]],
) -> dict[str, float]:
    """Average the values of target."""
    return {key: sum(value) / len(value) for key, value in target.items()}

def log_dict(
    target: dict[str, float],
    epoch: int,
    prefix: str,
) -> None:
    """Log the values of target."""
    for key, value in target.items():
        logger.log_to_external(
                message={
                    f"{prefix}/{key}": value,
                    "Epoch": epoch,
                },
            )