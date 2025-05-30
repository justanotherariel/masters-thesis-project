"""Example transformation block for the transformation pipeline."""

import numpy as np
import numpy.typing as npt

from src.framework.transforming import TransformationBlock


class ExampleTransformationBlock(TransformationBlock):
    """An example transformation block for the transformation pipeline."""

    def custom_transform(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        return data
