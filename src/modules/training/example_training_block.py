"""Module for example training block."""

import numpy as np
import numpy.typing as npt

from src.framework.training_block import TrainingBlock


class ExampleTrainingBlock(TrainingBlock):
    """An example training block."""

    def train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        # ("Training")
        return x, y

    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Predict using the model.

        :param x: The input data
        :return: The predictions
        """
        # ("Predicting")
        return x
