from torch import nn
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo

from .base import BaseModel
from .unet import UNet


class UNetWrappedAgentOnly(BaseModel):
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

        return f"UNet({', '.join(attrs)})"

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._info = info
        self._ti = info.model_ti

        # Set observation shape and action dimension from environment info
        obs_shape = (*info.data_info["observation_space"].shape[:-1], self._ti.observation_.shape[0])
        action_dim = info.data_info["action_space"].n.item()

        self.module = UNet(
            in_obs_shape=obs_shape,
            out_obs_shape=(*obs_shape[:-1], len(self._ti.observation[3])),
            action_dim=action_dim,
            **self._model_args,
        )

        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]

        return info

    @staticmethod
    def get_dataset_cls():
        from ..datasets.simple import SimpleDatasetDefault

        return SimpleDatasetDefault

    def forward(self, x):
        # Forward pass through the model
        pred_obs_agent, pred_reward = self.module.forward(x)
        
        pred_obs = x[0].float()
        pred_obs[..., self._ti.observation[3]] = pred_obs_agent.reshape(*pred_obs_agent.shape[:-1], -1)

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
