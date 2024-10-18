"""TransformationBlock module than can be extended by implementing the custom_transform method."""

from abc import abstractmethod
from typing import Any

from .logging import Logger
from .transforming import Transformer
from .caching import CacheArgs, Cacher

logger = Logger()

class TransformationBlock(Transformer, Cacher):
    """The transformation block is a flexible block that allows for transformation of any data.

    Methods
    -------
    .. code-block:: python
        @abstractmethod
        def custom_transform(self, data: Any, **transform_args) -> Any: # Custom transformation implementation

        def transform(self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any) -> Any: # Applies caching and calls custom_transform

    Usage:
    .. code-block:: python
        from src.framework.transformation_block import TransformationBlock

        class CustomTransformationBlock(TransformationBlock):
            def custom_transform(self, data: Any) -> Any:
                return data

            ....

        custom_transformation_block = CustomTransformationBlock()

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "data/processed",
        }

        data = custom_transformation_block.transform(data, cache=cache_args)
    """

    def transform(self, data: Any, cache_args: CacheArgs | None = None, **transform_args: Any) -> Any:  # noqa: ANN401
        """Transform the input data using a custom method.

        :param data: The input data.
        :param cache_args: The cache arguments.
        :return: The transformed data.
        """
        if cache_args and self.cache_exists(
            name=self.get_hash(),
            cache_args=cache_args,
        ):
            logger.info(
                f"Cache exists for {self.__class__} with hash: {self.get_hash()}. Using the cache.",
            )
            return self._get_cache(name=self.get_hash(), cache_args=cache_args)

        data = self.custom_transform(data, **transform_args)
        if cache_args:
            logger.info(f"Storing cache to {cache_args['storage_path']}")
            self._store_cache(name=self.get_hash(), data=data, cache_args=cache_args)
        return data

    @abstractmethod
    def custom_transform(self, data: Any, **transform_args: Any) -> Any:  # noqa: ANN401
        """Transform the input data using a custom method.

        :param data: The input data.
        :return: The transformed data.
        """
        raise NotImplementedError(
            f"Custom transform method not implemented for {self.__class__}",
        )
