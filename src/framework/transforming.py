from abc import abstractmethod
from typing import Any

from src.framework.caching import CacheArgs, Cacher
from src.framework.logging import Logger

from .core import _Base, _Block, _SequentialSystem

logger = Logger()


class TransformType(_Base):
    """Abstract transform type describing a class that implements the transform function"""

    @abstractmethod
    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :param transform_args: Keyword arguments.
        :return: The transformed data."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement transform method.")


class Transformer(TransformType, _Block):
    """The transformer block transforms any data it could be x or y data.

    Methods:
    .. code-block:: python
        @abstractmethod
        def transform(self, data: Any, **transform_args: Any) -> Any:
            # Transform the input data.

        def get_hash(self) -> str:
            # Get the hash of the Transformer

        def get_parent(self) -> Any:
            # Get the parent of the Transformer

        def get_children(self) -> list[Any]:
            # Get the children of the Transformer

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from src.framework.transforming import Transformer

        class MyTransformer(Transformer):
            def transform(self, data: Any, **transform_args: Any) -> Any:
                # Transform the input data.
                return data

        my_transformer = MyTransformer()
        transformed_data = my_transformer.transform(data)
    """


class TransformationBlock(Transformer, Cacher):
    """The transformation block is a flexible block that allows for transformation of any data.

    Methods
    -------
    .. code-block:: python
        @abstractmethod
        def custom_transform(self, data: Any, **transform_args) -> Any: # Custom transformation implementation

        def transform(self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any) -> Any:
        # Applies caching and calls custom_transform

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


class TransformingSystem(TransformType, _SequentialSystem):
    """A system that transforms the input data.

    Parameters:
    - steps (list[Transformer | TransformingSystem | ParallelTransformingSystem]): The steps in the system.

    Implements the following methods:
    .. code-block:: python
        def transform(self, data: Any, **transform_args: Any) -> Any:
            # Transform the input data.

        def get_hash(self) -> str:
            # Get the hash of the TransformingSystem

        def get_parent(self) -> Any:
            # Get the parent of the TransformingSystem

        def get_children(self) -> list[Any]:
            # Get the children of the TransformingSystem

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path


    Usage:
    .. code-block:: python
        from src.framework.transforming import TransformingSystem

        transformer_1 = CustomTransformer()
        transformer_2 = CustomTransformer()

        transforming_system = TransformingSystem(steps=[transformer_1, transformer_2])
        transformed_data = transforming_system.transform(data)
        predictions = transforming_system.predict(data)
    """

    def __post_init__(self) -> None:
        """Post init method for the TransformingSystem class."""

        # Assert all steps are a subclass of Transformer
        for step in self.steps:
            if not isinstance(step, (TransformType)):
                raise TypeError(f"{step} is not an instance of TransformType")

        super().__post_init__()

    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """

        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)
        if set_of_steps != set(transform_args.keys()):
            # Raise a warning and print all the keys that do not match
            logger.warning(
                "The following steps do not exist but were given in the kwargs: "
                f"{set(transform_args.keys()) - set_of_steps}"
            )

        # Loop through each step and call the transform method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = transform_args.get(step_name, {})
            if isinstance(step, (TransformType)):
                data = step.transform(data, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TransformType")

        return data
