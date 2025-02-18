from torch import nn

from src.typing.pipeline_objects import PipelineInfo


class BaseModel:
    module: nn.Module

    def __init__(
        self,
        **kwargs,
    ):
        raise NotImplementedError("BaseModel is an abstract class and should not be instantiated.")

    def setup(self, info):
        raise NotImplementedError("BaseModel is an abstract class and should not be instantiated.")

    @staticmethod
    def get_dataset_cls():
        raise NotImplementedError("BaseModel is an abstract class and should not be called.")

    def forward(self, x):
        raise NotImplementedError("BaseModel is an abstract class and should not be called.")


class MinigridModel:
    module: None | nn.Module

    def __init__(
        self,
        model_cls,
        **model_args,
    ):
        self.model_cls = model_cls
        self.name = model_cls.func.__name__

        # Transformer parameters
        self._model_args = model_args

    def __repr__(self):
        attrs = []
        for attr, attr_val in self._model_args.items():
            attrs.append(f"{attr}={attr_val}")

        return f"{self.name}({', '.join(attrs)})"

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._info = info
        self._ti = info.model_ti

        # Set observation shape and action dimension from environment info
        obs_shape = (*info.data_info["observation_space"].shape[:-1], self._ti.observation_.shape[0])
        action_dim = info.data_info["action_space"].n.item()

        self.module = self.model_cls(
            in_obs_shape=obs_shape,
            out_obs_shape=obs_shape,
            action_dim=action_dim,
            **self._model_args,
        )

        return info

    @staticmethod
    def get_dataset_cls():
        from ..datasets.simple import SimpleDatasetDefault

        return SimpleDatasetDefault

    def forward(self, x):
        return self.module.forward(x)
