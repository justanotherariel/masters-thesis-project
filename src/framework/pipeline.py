"""Model module. Contains the ModelPipeline class."""

from typing import Any
from joblib import hash
from dataclasses import dataclass

from .transforming import TransformType, Transformer, TransformingSystem
from .caching import Cacher, CacheArgs
from src.framework.logging import Logger
from src.typing.pipeline_objects import XData

logger = Logger()


@dataclass
class ModelPipeline(TransformType):
    """ModelPipeline is the class used to create the pipeline for the model.

    :param env_sys: Setup the environment for the pipeline.
    :param train_sys: The system to train the model.
    :param pred_sys: The system to predict the output.
    """

    env_sys: TransformingSystem | None = None
    train_sys: Transformer | TransformingSystem | None = None
    pred_sys: TransformingSystem | None = None

    def __post_init__(self) -> None:
        """Post initialization function of the Pipeline."""
        super().__post_init__()

        # Set children and parents
        children = []
        systems = [
            self.env_sys,
            self.train_sys,
            self.pred_sys,
        ]

        for sys in systems:
            if sys is not None:
                sys._set_parent(self)
                children.append(sys)

        self._set_children(children)

    def transform(self, **transform_args: Any) -> XData:
        """Train the system.

        :param train_args: The arguments to pass to the training system. (Default is {})
        :return: The output of the system.
        """
        data = None
        if self.env_sys is not None:
            data = self.env_sys.transform(data, **transform_args.get("env_sys", {}))
        if self.train_sys is not None:
            data, y = self.train_sys.transform(
                data, **transform_args.get("train_sys", {})
            )
        if self.pred_sys is not None:
            data = self.pred_sys.transform(data, **transform_args.get("pred_sys", {}))

        return data

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the pipeline.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        if self.env_sys is not None:
            self.env_sys._set_hash(self.get_hash())
            env_hash = self.env_sys.get_hash()
            if env_hash != "":
                self._hash = hash(self._hash + env_hash)

        if self.train_sys is not None:
            self.train_sys._set_hash(self.get_hash())
            train_hash = self.train_sys.get_hash()
            if train_hash != "":
                self._hash = hash(self._hash + train_hash)

        if self.pred_sys is not None:
            self.pred_sys._set_hash(self.get_hash())
            pred_hash = self.pred_sys.get_hash()
            if pred_hash != "":
                self._hash = hash(self._hash + pred_hash)

    def get_env_cache_exists(self, cache_args: CacheArgs) -> bool:
        """Get status of env.

        :param cache_args: Cache arguments
        :return: Whether cache exists
        """
        if self.env_sys is None:
            return False
        return self.env_sys.cache_exists(self.env_sys.get_hash(), cache_args)


@dataclass
class TransformationPipeline(TransformingSystem, Cacher):
    """TransformationPipeline is the class used to create the pipeline for the transformation of the data.

    ### Parameters:
    - `steps` (List[Union[Transformer, TransformationPipeline]]): The steps to transform the data. Can be a list of any Transformer type.
    - `title` (str): The title of the pipeline. (Default: "Transformation Pipeline")

    Methods
    -------
    .. code-block:: python
        def transform(self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any) -> Any: # Transform the input data.

        def get_hash(self) -> str: # Get the hash of the pipeline.

    Usage:
    .. code-block:: python
        from pipline import TransformationPipeline

        class MyTransformationPipeline(TransformationPipeline):
            ...

        step1 = MyTransformer1()
        step2 = MyTransformer2()
        pipeline = MyTransformationPipeline(steps=[step1, step2])

        data = pipeline.transform(data)
    """

    title: str = "Transformation Pipeline"  # The title of the pipeline since transformation pipeline can be used for multiple purposes. (Feature, Label, etc.)

    def transform(
        self, data: Any, cache_args: CacheArgs | None = None, **transform_args: Any
    ) -> Any:  # noqa: ANN401
        """Transform the input data.

        :param data: The input data.
        :param cache_args: The cache arguments.
        :return: The transformed data.
        """
        if cache_args and self.cache_exists(self.get_hash(), cache_args):
            logger.info(
                f"Cache exists for {self.title} with hash: {self.get_hash()}. Using the cache.",
            )
            return self._get_cache(self.get_hash(), cache_args)

        if self.get_steps():
            logger.log_section_separator(self.title)

        self.all_steps = self.get_steps()

        # Furthest step
        for i, step in enumerate(self.get_steps()):
            # Check if step is instance of Cacher and if cache_args exists
            if not isinstance(step, Cacher) or not isinstance(step, TransformType):
                logger.debug(f"{step} is not instance of Cacher or TransformType")
                continue

            step_args = transform_args.get(step.__class__.__name__, None)
            if step_args is None:
                logger.debug(f"{step} is not given transform_args")
                continue

            step_cache_args = step_args.get("cache_args", None)
            if step_cache_args is None:
                logger.debug(f"{step} is not given cache_args")
                continue

            step_cache_exists = step.cache_exists(step.get_hash(), step_cache_args)
            if step_cache_exists:
                logger.debug(
                    f"Cache exists for {step}, moving index of steps to {i}",
                )
                self.steps = self.all_steps[i:]

        data = super().transform(data, **transform_args)

        if cache_args:
            logger.info(f"Storing cache for pipeline to {cache_args['storage_path']}")
            self._store_cache(self.get_hash(), data, cache_args)

        # Set steps to original in case class is called again
        self.steps = self.all_steps

        return data
