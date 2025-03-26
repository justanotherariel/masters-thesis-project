import copy
from dataclasses import dataclass, field
from typing import Annotated, Any

from annotated_types import Gt

from src.framework.logging import Logger
from src.typing.pipeline_objects import DatasetGroup

from ..models.base import BaseModel

logger = Logger()


@dataclass
class EarlyStopping:
    enable: bool = field(default=False)
    start_epoch: Annotated[int, Gt(0)] = field(default=0)
    patience: Annotated[int, Gt(0)] = field(default=int(1e6))
    metric: tuple[DatasetGroup, str] = field(default=("VALIDATION", "Loss"))
    metric_mode: str = field(default="min")  # min or max
    metric_min_delta: float = field(default=0.0)
    revert_to_best_model: bool = field(default=True)

    def __post_init__(self):
        self.metric = (DatasetGroup[self.metric[0]], self.metric[1])
        
        self.counter = 0
        self.best_score = float("inf") if self.metric_mode == "min" else -float("inf")
        self.best_model_dict: dict[str, Any] | None = None

    def __call__(self, epoch: int, metrics: dict[DatasetGroup, dict[str, float]], model: BaseModel) -> bool:
        """Should run after every epoch. Returns True if the early stopping condition is met.

        Args:
            epoch (int): Current Epoch
            metrics (dict[DatasetGroup, dict[str, float]]): Metrics Collected

        Returns:
            bool: Stopping Condition Met
        """

        if not self.enable:
            return False

        if epoch < self.start_epoch:
            return False

        # Check if the metric is present - Validation might not always be performed
        is_best_score = False
        is_best_score_delta = False
        if self.metric[0] in metrics and self.metric[1] in metrics[self.metric[0]]:
            score = metrics[self.metric[0]][self.metric[1]]
            
            if self.metric_mode == "min":
                is_best_score = score < self.best_score
                is_best_score_delta = score + self.metric_min_delta < self.best_score
            else:
                is_best_score = score > self.best_score
                is_best_score_delta = score - self.metric_min_delta > self.best_score

            # If the score is better
            if is_best_score:
                # Update the best score
                self.best_model_score = score

                # Store the best model
                if self.revert_to_best_model:
                    self.best_model_dict = copy.deepcopy(model.module.state_dict())

        # If the score incl delta is better
        if is_best_score_delta:
            self.best_score = score
            
            # Reset the counter
            self.counter = 0
        
        # If the score incl delta is worse
        else:
            # Increment the counter
            self.counter += 1

            # Check if we should stop and return True - early stopping
            if self.counter >= self.patience:
                logger.info(f"Early Stopping! \n"
                            f"Metric: {self.metric[0]}/{self.metric[1]}\n"
                            f"Best Score: {self.best_model_score}\n")
                return True
        return False

    def load_best_model(self, model: BaseModel):
        """Should be called after the end of training to load the best model.

        Args:
            model (BaseModel): The model to load the best weights into.
        """

        if self.best_model_dict is not None:
            logger.info(f"Early stopping and revert_to_best_model enabled - reverting model.")
            model.module.load_state_dict(self.best_model_dict)
