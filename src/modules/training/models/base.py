from torch import nn


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
