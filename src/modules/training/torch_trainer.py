"""TorchTrainer is a module that allows for the training of Torch models."""

import contextlib
import copy
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from annotated_types import Ge, Gt, Interval
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from src.framework.logging import Logger
from src.framework.trainers.data_parallel import DataParallel
from src.framework.trainers.utils import _get_onnxrt, _get_openvino
from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import XData

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
class TorchTrainer(TransformationBlock):
    """Abstract class for torch trainers, override necessary functions for custom implementation.

    Parameters Modules
    ----------
    - `model` (nn.Module): The model to train.
    - `optimizer` (functools.partial[Optimizer]): Optimizer to use for training.
    - `criterion` (nn.Module): Criterion to use for training.
    - `scheduler` (Callable[[Optimizer], LRScheduler] | None): Scheduler to use for training.
    - `dataloader_args (dict): Arguments for the dataloader`

    Parameters Training
    ----------
    - `epochs` (int): Number of epochs
    - `patience` (int): Stopping training after x epochs of no improvement (early stopping)
    - `batch_size` (int): Batch size

    Parameters Checkpointing
    ----------
    - `checkpointing_enabled` (bool): Whether to save checkpoints after each epoch
    - `checkpointing_keep_every` (int): Keep every i'th checkpoint (1 to keep all, 0 to keep only the last one)
    - `checkpointing_resume_if_exists` (bool): Resume training if a checkpoint exists

    Parameters Precision
    ----------
    - `use_mixed_precision` (bool): Whether to use mixed precision for the model training

    Parameters Misc
    ----------
    - `to_predict` (str): Whether to predict on the 'validation' set, 'all' data or 'none'
    - `model_name` (str): Name of the model
    - `n_folds` (float): Number of folds for cross validation (0 for train full,
    - `_fold` (int): Fold number
    - `validation_size` (float): Relative size of the validation set
    """

    # Modules
    model: nn.Module
    optimizer: functools.partial[Optimizer]
    criterion: nn.Module
    scheduler: Callable[[Optimizer], LRScheduler] | None = None
    dataloader_args: dict[str, Any] = field(default_factory=dict, repr=False)

    # Training parameters
    epochs: Annotated[int, Gt(0)] = 10
    patience: Annotated[int, Gt(0)] = -1  # Early stopping
    batch_size: Annotated[int, Gt(0)] = 32
    collate_fn: Callable[[list[Tensor]], tuple[Tensor, Tensor]] = field(
        default=custom_collate, init=True, repr=False, compare=False
    )

    # Checkpointing
    checkpointing_enabled: bool = field(default=True, init=True, repr=False, compare=False)
    checkpointing_keep_every: Annotated[int, Gt(0)] = field(default=0, init=True, repr=False, compare=False)
    checkpointing_resume_if_exists: bool = field(default=True, init=True, repr=False, compare=False)

    # Precision
    use_mixed_precision: bool = field(default=False)

    # Misc
    model_name: str | None = None  # No spaces allowed
    trained_models_directory: PathLike[str] = field(default=Path("tm/"), repr=False, compare=False)
    to_predict: Literal["validation", "all", "none"] = field(default="validation", repr=False, compare=False)
    setup_info: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    # Parameters relevant for Hashing
    n_folds: Annotated[int, Ge(0)] = field(default=-1, init=True, repr=False, compare=False)
    _fold: int = field(default=-1, init=False, repr=False, compare=False)
    validation_size: Annotated[float, Interval(ge=0, le=1)] = 0.2

    # Prefix and postfix for logging to external
    logging_prefix: str = field(default="", init=True, repr=False, compare=False)
    logging_postfix: str = field(default="", init=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post init method for the TorchTrainer class."""
        # Make sure to_predict is either "validation" or "all" or "none"
        if self.to_predict not in ["validation", "all", "none"]:
            raise ValueError("to_predict should be either 'validation', 'all' or 'none'")

        if self.n_folds == -1:
            raise ValueError(
                "Please specify the number of folds for cross validation or set n_folds to 0 for train full.",
            )

        if self.model_name is None:
            raise ValueError("self.model_name is None, please specify a model_name")

        # Check validity of model_name
        if " " in self.model_name:
            raise ValueError("Spaces in model_name not allowed")

        self.save_model_to_disk = True
        self.best_model_state_dict: dict[Any, Any] = {}

        # Set optimizer
        self.initialized_optimizer = self.optimizer(self.model.parameters())

        # Set scheduler
        self.initialized_scheduler: LRScheduler | None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)
        else:
            self.initialized_scheduler = None

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting device: {self.device}")

        # If multiple GPUs are available, distribute batch size over the GPUs
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(self.model)

        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self._lowest_val_loss = np.inf

        # Mixed precision
        if self.use_mixed_precision:
            logger.info("Using mixed precision training.")
            self.scaler = torch.GradScaler(device=self.device.type)
            torch.set_float32_matmul_precision("high")

        super().__post_init__()

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param data: The input data.
        :return: The transformed data.
        """
        self.setup_info = info

        if isinstance(self.model.get_dataset_cls(), functools.partial):
            info["token_index"] = self.model.get_dataset_cls().func.create_ti(info)
        else:
            info["token_index"] = self.model.get_dataset_cls().create_ti(info)

        self.model.setup(info)

        return info

    def custom_transform(self, data: XData, **train_args: Any) -> XData:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments.
            - train_indices: The indices to train on.
            - validation_indices: The indices to validate on.
            - save_model: Whether to save the model.
            - fold: Fold number if running cv.
        :return: The input and output of the system.
        """
        self._fold = train_args.get("fold", -1)

        # Train or load model from disk
        if self._model_exists():
            logger.info(
                f"Model exists in {self.get_model_path()}. Loading model...",
            )
            self._load_model()
        else:
            self._model_train(data)

        # Evaluate the model
        loader = self.create_dataloader(data, f"{self.to_predict}_indices", shuffle=False)
        data.validation_predictions, data.validation_labels = self.predict_on_loader(loader)
        return data

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
        compile_method: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        logger.info("Running inference on the given dataloader")
        self.model.eval()
        labels = []
        predictions = []
        if compile_method is None:
            with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = data[0].to(self.device)
                    y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
                    predictions.extend(y_pred)
                    labels.extend(data[1])

        elif compile_method == "ONNX":
            if self.device != torch.device("cpu"):
                raise ValueError(
                    "ONNX compilation only works on CPU. To disable CUDA use the "
                    "environment variable CUDA_VISIBLE_DEVICES=-1"
                )
            input_tensor = next(iter(loader))[0].to(self.device).float()
            input_names = ["actual_input"]
            output_names = ["output"]
            logger.info("Compiling model to ONNX")
            torch.onnx.export(
                self.model,
                input_tensor,
                f"{self.get_hash()}.onnx",
                verbose=False,
                input_names=input_names,
                output_names=output_names,
            )
            onnx_model = _get_onnxrt().InferenceSession(f"{self.get_hash()}.onnx")
            logger.info("Done compiling model to ONNX")
            with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = data[0].to(self.device)
                    y_pred = onnx_model.run(output_names, {input_names[0]: X_batch.numpy()})[0]
                    predictions.extend(y_pred)
                    labels.extend(data[1])

            # Remove the onnx file
            Path(f"{self.get_hash()}.onnx").unlink()

        elif compile_method == "Openvino":
            if self.device != torch.device("cpu"):
                raise ValueError(
                    "Openvino compilation only works on CPU. To disable CUDA use the "
                    "environment variable CUDA_VISIBLE_DEVICES=-1"
                )
            input_tensor = next(iter(loader))[0].to(self.device).float()
            logger.info("Compiling model to Openvino")
            ov = _get_openvino()
            openvino_model = ov.compile_model(ov.convert_model(self.model, example_input=input_tensor))
            logger.info("Done compiling model to Openvino")
            with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = data[0].to(self.device)
                    y_pred = openvino_model(X_batch)[0]
                    predictions.extend(y_pred)
                    labels.extend(data[1])

        logger.info("Done predicting!")
        return np.array(predictions), np.array(labels)

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        result = f"{self._hash}_{self.n_folds}"
        if self._fold != -1:
            result += f"_f{self._fold}"
        return result

    def create_dataloader(
        self,
        data: XData,
        indices: str,
        shuffle: bool = True,
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        # Create datasets
        dataset = self.model.get_dataset_cls()(data, indices)
        dataset.setup(self.setup_info)

        # Create dataloaders
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=(self.collate_fn if hasattr(dataset, "__getitems__") else None),
            **self.dataloader_args,
        )
        return loader

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self.trained_models_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def _model_train(
        self,
        data: XData,
    ):
        # Log the model being trained
        logger.info(f"Training model: {self.model.__class__.__name__}")

        # Create dataloaders
        train_loader = self.create_dataloader(data, "train_indices")
        validation_loader = self.create_dataloader(data, "validation_indices")

        # Resume from checkpoint if enabled and checkpoint exists
        start_epoch = 0
        if self.checkpointing_resume_if_exists:
            saved_checkpoints = list(Path(self.trained_models_directory).glob(f"{self.get_hash()}_checkpoint_*.pt"))
            if len(saved_checkpoints) > 0:
                logger.info("Resuming training from checkpoint")
                epochs = [int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints]
                self._load_model(saved_checkpoints[np.argmax(epochs)])
                start_epoch = max(epochs) + 1

        # Train the model
        logger.info(
            f"Training model for {self.epochs} epochs"
            f"{', starting at epoch ' + str(start_epoch) if start_epoch > 0 else ''}"
        )
        self._lowest_val_loss = np.inf
        self._model_training_loop(
            train_loader,
            validation_loader,
            self._fold,
            start_epoch,
        )
        logger.info(
            f"Done training the model: {self.model.__class__.__name__}",
        )

        # Revert to the best model
        if self.best_model_state_dict:
            logger.info(
                f"Reverting to model with best validation loss {self._lowest_val_loss}",
            )
            self.model.load_state_dict(self.best_model_state_dict)

        # Save the model
        if self.save_model_to_disk:
            self._save_model()

    def _model_training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        validation_loader: DataLoader[tuple[Tensor, ...]],
        fold: int = -1,
        start_epoch: int = 0,
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the validation data.
        :param validation_loader: Dataloader for the training data. (can be empty)
        """
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

        logger.external_define_metric(self.wrap_log(f"Training/Train Loss{fold_no}"), self.wrap_log("epoch"))
        logger.external_define_metric(
            self.wrap_log(f"Validation/Validation Loss{fold_no}"),
            self.wrap_log("epoch"),
        )

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(start_epoch, self.epochs):
            # Train using train_loader
            train_loss = self.train_one_epoch(train_loader, epoch)
            logger.debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            logger.log_to_external(
                message={
                    self.wrap_log(f"Training/Train Loss{fold_no}"): train_losses[-1],
                    self.wrap_log("epoch"): epoch,
                },
            )

            # Step the scheduler
            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.checkpointing_enabled:
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
            if len(validation_loader) > 0:
                self.last_val_loss = self.val_one_epoch(
                    validation_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                logger.debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
                logger.log_to_external(
                    message={
                        self.wrap_log(f"Validation/Validation Loss{fold_no}"): val_losses[-1],
                        self.wrap_log("epoch"): epoch,
                    },
                )

                logger.log_to_external(
                    message={
                        "type": "wandb_plot",
                        "plot_type": "line_series",
                        "data": {
                            "xs": list(
                                range(epoch + 1),
                            ),  # Ensure it's a list, not a range object
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": self.wrap_log(f"Training/Loss{fold_no}"),
                            "xname": "Epoch",
                        },
                    },
                )

                # Early stopping
                if self._early_stopping():
                    logger.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): (epoch + 1) - self.patience})
                    break

            # Log the trained epochs to wandb if we finished training
            logger.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): epoch + 1})

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
        )
        for batch in pbar:
            x_batch, y_batch = batch

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model(x_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch.float())

            # Backward pass
            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Remove the cuda cache
        # torch.cuda.empty_cache()
        # gc.collect()

        return sum(losses) / len(losses)

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                x_batch, y_batch = batch

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model(x_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch.float())

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def _save_model(
        self,
        model_path: Path | None = None,
        *,
        save_to_external: bool = True,
        quiet: bool = False,
    ) -> None:
        """Save the model in the model_directory folder."""
        model_path = model_path if model_path is not None else self.get_model_path()

        if not quiet:
            logger.info(
                f"Saving model to {model_path}",
            )

        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, model_path)

        if save_to_external:
            self.save_model_to_external()

    def _load_model(self, path: Path | None = None) -> None:
        """Load the model from the model_directory folder."""
        model_path = path if path is not None else self.get_model_path()

        # Check if the model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found in {model_path}",
            )

        # Load model
        logger.info(
            f"Loading model from {model_path}",
        )
        checkpoint = torch.load(model_path, weights_only=False)

        # Load the weights from the checkpoint
        model = checkpoint.module if isinstance(checkpoint, nn.DataParallel) else checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

    def _model_exists(self) -> bool:
        """Check if the model exists in the model_directory folder."""
        return self.get_model_path().exists() and self.save_model_to_disk

    def _early_stopping(self) -> bool:
        """Check if early stopping should be performed.

        :return: Whether to perform early stopping.
        """
        # Store the best model so far based on validation loss
        if self.patience != -1:
            if self.last_val_loss < self._lowest_val_loss:
                self._lowest_val_loss = self.last_val_loss
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    logger.info(
                        f"Early stopping after {self.early_stopping_counter} epochs",
                    )
                    return True
        return False

    def get_model_path(self) -> Path:
        """Get the model path.

        :return: The model path.
        """
        return Path(self.trained_models_directory) / f"{self.get_hash()}.pt"

    def get_model_checkpoint_path(self, epoch: int) -> Path:
        """Get the checkpoint path.

        :param epoch: The epoch number.
        :return: The checkpoint path.
        """
        return Path(self.trained_models_directory) / f"{self.get_hash()}_checkpoint_{epoch}.pt"

    def wrap_log(self, text: str) -> str:
        """Add logging prefix and postfix to the message."""
        return f"{self.logging_prefix}{text}{self.logging_postfix}"
