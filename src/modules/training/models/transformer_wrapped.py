import math

import torch
from torch import nn
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo

from .base import BaseModel
from .transformer import Transformer


class TransformerWrapped(BaseModel):
    module: None | nn.Module

    def __init__(
        self,
        **model_args,
    ):
        # Transformer parameters
        self._model_args = model_args

    def __repr__(self):
        attrs = []
        for attr, attr_val in self._model_args.items():
            attrs.append(f"{attr}={attr_val}")

        return f"Transformer({', '.join(attrs)})"

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._info = info
        self._ti = info.model_ti

        # Set observation shape and action dimension from environment info
        self.obs_shape = (*info.data_info["observation_space"].shape[:-1], self._ti.observation_.shape[0])
        self._token_input_dim: tuple[int, int] = (
            math.prod(self.obs_shape[:2]) + 1,
            self.obs_shape[2] + self._ti.action_.shape[0],
        )
        self._token_output_dim: tuple[int, int] = (
            math.prod(self.obs_shape[:2]) + 1,
            self.obs_shape[2] + self._ti.reward_.shape[0],
        )

        self.module = Transformer(
            input_dim=self._token_input_dim,
            output_dim=self._token_output_dim,
            **self._model_args,
        )

        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]

        return info

    @staticmethod
    def get_dataset_cls():
        from ..datasets.simple import SimpleDatasetDefault

        return SimpleDatasetDefault

    def forward(self, x):
        x_obs, x_action = x
        samples = x_obs.shape[0]

        # Reshape the input to a token sequence
        x = torch.zeros(samples, *self._token_input_dim, dtype=torch.uint8, device=x_obs.device)
        x[:, :-1, self._ti.observation_] = x_obs.view(samples, math.prod(self.obs_shape[:2]), self.obs_shape[2])
        x[:, -1, self._ti.action_] = x_action.view(samples, self._ti.action_.shape[0])

        # Forward pass through the transformer
        x = self.module.forward(x)

        # Reshape the output to a model-independent format
        pred_obs = x[:, :-1, self._ti.observation_].reshape(*x_obs.shape[:-1], self.obs_shape[2])
        pred_reward = x[:, -1, self._ti.reward_]

        # Softmax the observation
        for values in self._tensor_values:
            # Only apply softmax if range has multiple elements
            if len(values) > 1:
                # Extract the relevant slice
                sliced = pred_obs[..., values]

                # Apply softmax along the last dimension
                softmaxed = F.softmax(sliced, dim=-1)

                # Place back in output
                pred_obs[..., values] = softmaxed

        return pred_obs, pred_reward
