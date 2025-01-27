from typing import Any

import torch

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import PipelineData, PipelineInfo

logger = Logger()


def convert_token_dataset(data: PipelineData, info: PipelineInfo) -> Any:
    def convert_tokens(
        data: torch.Tensor, ti: TokenIndex, obs_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_tensor = torch.stack(data)
        obs = data_tensor[..., :-1, :].reshape(data_tensor.shape[0], *obs_shape, -1)
        reward = data_tensor[..., -1, ti.reward_].float()
        return obs, reward

    ti = info.model_ti
    shape = info.data_info["observation_space"].shape[:2]

    for key in data.predictions:
        data.predictions[key] = convert_tokens(data.predictions[key], ti, shape)

    return data


def convert_two_d_dataset(data: PipelineData, info: PipelineInfo) -> Any:
    return data


class MinigridPredictionReshape(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        return info

    def custom_transform(self, data: PipelineData, **kwargs) -> PipelineData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """

        CONVERT = {"TokenDataset": convert_token_dataset, "TwoDDataset": convert_two_d_dataset}

        return CONVERT[self.info.model_ds_class](data, self.info)
