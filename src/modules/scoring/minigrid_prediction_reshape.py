from typing import Any

import torch

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import XData

logger = Logger()


def convert_token_dataset(data: XData, info: dict) -> Any:
    def convert_tokens(
        data: torch.Tensor, ti: TokenIndex, obs_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_tensor = torch.stack(data)
        obs = data_tensor[..., :-1, :].reshape(data_tensor.shape[0], *obs_shape, -1)
        reward = data_tensor[..., -1, ti.reward_].float()
        return obs, reward

    ti = info["token_index"]
    shape = info["env_build"]["observation_space"].shape[:2]

    if data.train_predictions is not None:
        data.train_predictions = convert_tokens(data.train_predictions, ti, shape)

    if data.train_targets is not None:
        data.train_targets = convert_tokens(data.train_targets, ti, shape)

    if data.validation_predictions is not None:
        data.validation_predictions = convert_tokens(data.validation_predictions, ti, shape)

    if data.validation_targets is not None:
        data.validation_targets = convert_tokens(data.validation_targets, ti, shape)

    return data

def convert_two_d_dataset(data: XData, info: dict) -> Any:
    return data


class MinigridPredictionReshape(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        return info

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """

        CONVERT = {
            "TokenDataset": convert_token_dataset, 
            "TwoDDataset": convert_two_d_dataset
        }

        return CONVERT[self.info["train"]["dataset"]](data, self.info)
