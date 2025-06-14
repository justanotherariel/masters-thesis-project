"""TorchTrainer is a module that allows for the training of Torch models."""

import contextlib
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from annotated_types import Ge, Gt
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .accuracy import BaseAccuracy
from .loss import BaseLoss
from .models.base import BaseModel
from .utils.early_stopping import EarlyStopping
from .utils.model_storage import ModelStorage

logger = Logger()


@dataclass
class ModelStorageConf:
    save_model_to_disk: bool = field(default=True, repr=False, compare=False)
    save_model_to_wandb: bool = field(default=True, repr=False, compare=False)

    save_checkpoints_to_disk: bool = field(default=True, repr=False, compare=False)
    save_checkpoints_keep_every_x_epochs: Annotated[int, Gt(0)] = field(default=0, repr=False, compare=False)
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
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    dataloader_conf: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # Training parameters
    epochs: Annotated[int, Gt(0)] = 10
    batch_size: Annotated[int, Gt(0)] = 32
    use_mixed_precision: bool = field(default=False)
    validate_every_x_epochs: Annotated[int, Gt(0)] = 1
    load_all_batches_to_gpu: bool = field(default=False)  # Load and keep all batches in GPU memory
    discrete: bool = field(default=True)  # Discretize the data before training

    # Predction parameters
    to_predict: DatasetGroup = field(default="ALL", repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post init method for the TorchTrainer class."""

        # Setup ModelStorage
        self.model_storage_conf = ModelStorageConf(**self.model_storage_conf)
        ms = self.model_storage_conf
        if ms.save_model_to_wandb and not ms.save_model_to_disk:
            raise ValueError("Cannot save model to wandb without saving to disk.")

        # Setup EarlyStopping
        self.early_stopping = EarlyStopping(**self.early_stopping) if isinstance(self.early_stopping, dict) else self.early_stopping

        # Convert to_predict string to DatasetGroup
        self.to_predict = DatasetGroup[self.to_predict]

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
            data.model_training_time_s = time.time() - current_time
            mins = data.model_training_time_s // 60
            secs = data.model_training_time_s % 60
            min_str = "minutes" if mins != 1 else "minute"
            sec_str = "seconds" if secs != 1.0 else "second"
            logger.info(f"Training took {f'{mins} {min_str} and ' if mins > 0 else ''}{secs:.2f} {sec_str}")

            data.model_last_epoch_recorded = self.last_epoch

        # Evaluate the model
        if DatasetGroup.TRAIN in self.to_predict:
            n_train_samples = sum([n_samples.shape[0] for n_samples in data.indices[DatasetGroup.TRAIN]])
            logger.info(f"Running inference on the training set. Samples: {n_train_samples}")
            logger.log_to_external({"Train Samples": n_train_samples})
            loader = self.create_dataloader(data, DatasetGroup.TRAIN, shuffle=False)
            data.predictions.update({DatasetGroup.TRAIN: self.predict_on_loader(loader)})

        if DatasetGroup.VALIDATION in self.to_predict and DatasetGroup.VALIDATION in data.indices:
            n_val_samples = sum([n_samples.shape[0] for n_samples in data.indices[DatasetGroup.VALIDATION]])
            logger.info(f"Running inference on the validation set. Samples: {n_val_samples}")
            logger.log_to_external({"Validation Samples": n_val_samples})
            loader = self.create_dataloader(data, DatasetGroup.VALIDATION, shuffle=False)
            data.predictions.update({DatasetGroup.VALIDATION: self.predict_on_loader(loader)})

        if DatasetGroup.TEST in self.to_predict and DatasetGroup.TEST in data.indices:
            n_test_samples = sum([n_samples.shape[0] for n_samples in data.indices[DatasetGroup.TEST]])
            logger.info(f"Running inference on the test set. Samples: {n_test_samples}")
            logger.log_to_external({"Test Samples": n_test_samples})
            loader = self.create_dataloader(data, DatasetGroup.TEST, shuffle=False)
            data.predictions.update({DatasetGroup.TEST: self.predict_on_loader(loader)})

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
        predictions = None

        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = moveTo(data[0], None, self.device)
                y_pred: dict[str, Tensor] = self.model.forward(X_batch)
                
                if predictions is None:
                    predictions = {key: [] for key in y_pred}
                
                for key in y_pred:
                    y_pred[key] = y_pred[key].to("cpu")
                for key, val in y_pred.items():
                    predictions[key].extend(val)
                    
            for key in predictions:
                predictions[key] = torch.stack(predictions[key])
                
        return predictions

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        result = f"{self._hash}"
        if hasattr(self, "current_fold"):
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
        logger.info(f"Training model: {self.model.name}")

        # Create dataloaders
        train_loader = self.create_dataloader(data, DatasetGroup.TRAIN)
        validation_loader = self.create_dataloader(data, DatasetGroup.TEST if DatasetGroup.TEST in data.indices else DatasetGroup.VALIDATION, shuffle=False)

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
        if self.early_stopping.revert_to_best_model:
            self.early_stopping.load_best_model(self.model)

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

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        # Track the number of epochs trained
        self.last_epoch = 0

        for epoch in range(start_epoch, self.epochs):
            # Train the model for one epoch
            train_metrics = self.train_one_epoch(train_loader, epoch)
            logger.debug(f"Epoch {epoch} Train Loss: {train_metrics['Loss']}")
            log_dict(train_metrics, "Train")

            # Step the scheduler
            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.model_storage_conf.save_checkpoints_to_disk:
                # Save checkpoint
                self.model_storage.save_model_checkpoint(
                    self.model.module,
                    epoch,
                )

                # Remove old checkpoints
                if (
                    self.model_storage_conf.save_checkpoints_keep_every_x_epochs == 0
                    or epoch % self.model_storage_conf.save_checkpoints_keep_every_x_epochs != 0
                ) and self.model_storage.get_model_checkpoint_path(epoch - 1).exists():
                    self.model_storage.get_model_checkpoint_path(epoch - 1).unlink()

            # Validate the model
            if len(validation_loader) > 0 and epoch % self.validate_every_x_epochs == 0:
                validation_metrics = self.val_one_epoch(validation_loader, epoch)
                logger.debug(f"Epoch {epoch} Valid Loss: {validation_metrics['Loss']}")
                log_dict(validation_metrics, "Test")

            # Log Epoch and commit Wandb Logs
            logger.log_to_external({"Epoch": epoch})
            self.last_epoch = epoch

            # Early stopping
            if self.early_stopping(
                epoch=epoch, metrics={DatasetGroup.TRAIN: train_metrics, DatasetGroup.VALIDATION: validation_metrics}, model=self.model
            ):
                break

        # Log the trained epochs to wandb after finishing training
        # Marks the true Epochs trained when early stopping is used
        logger.log_to_external(message={"Epochs": self.last_epoch + 1})

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
        epoch_loss: dict[str, list[float]] = {}
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
                loss, loss_dict = self.loss(y_pred, y_batch, x_batch)

            with torch.no_grad():
                acc_dict = self.accuracy(y_pred, y_batch, x_batch)

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
            append_to_dict(epoch_loss, loss_dict)
            append_to_dict(epoch_accuracy, acc_dict)
            pbar.set_postfix(loss=torch.cat(epoch_loss["Loss"]).mean().item())

        epoch_metrics = average_dict(epoch_loss)
        epoch_metrics.update(average_dict(epoch_accuracy))
        return epoch_metrics

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
        epoch_loss: dict[str, list[float]] = {}
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
                loss, loss_dict = self.loss(y_pred, y_batch, x_batch)
                acc_dict = self.accuracy(y_pred, y_batch, x_batch)

                # Save metrics
                append_to_dict(epoch_loss, loss_dict)
                append_to_dict(epoch_accuracy, acc_dict)
                pbar.set_postfix(loss=torch.cat(epoch_loss["Loss"]).mean().item())

        epoch_metrics = average_dict(epoch_loss)
        epoch_metrics.update(average_dict(epoch_accuracy))
        return epoch_metrics


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
    target: dict[str, list[Any]],
    source: dict[str, list[Any]],
) -> dict[str, Any]:
    """Append the values of source to target."""
    for key, value in source.items():
        if key not in target:
            target[key] = []
        target[key].append(value)
    return target


def average_dict(
    target: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    """Average the values of target."""
    result = {}
    for key, value in target.items():
        if isinstance(value[0], torch.Tensor):
            result[key] = torch.cat(value).float().mean().item()
        else:
            result[key] = np.mean(value)
    return result


def log_dict(
    target: dict[str, float],
    prefix: str,
    commit: bool = False,
) -> None:
    """Log the values of target."""
    message = {}
    for key, value in target.items():
        message[f"{prefix}/{key}"] = value
    logger.log_to_external(message, commit=commit)
