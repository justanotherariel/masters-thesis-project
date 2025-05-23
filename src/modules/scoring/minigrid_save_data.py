import pickle
from dataclasses import dataclass

from minigrid.core import actions

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.accuracy import obs_argmax
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

from .data_transform import dataset_to_list

logger = Logger()

ACTION_STR = [actions.Actions(i).name for i in range(7)]


@dataclass
class MinigridSaveData(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self._info = info
        return info

    def custom_transform(self, data: PipelineData, **kwargs) -> PipelineData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        logger.info("Saving data...")

        for dg in data.predictions:
            if dg == DatasetGroup.ALL:
                continue

            features, target = dataset_to_list(data, dg, discretize=False, info=self._info)
            pred_ti = self._info.model_ti
            pred_obs = obs_argmax(data.predictions[dg]["pred_obs"], pred_ti)
            predictions = (pred_obs, *data.predictions[dg].values())

            pickle.dump((features, target, predictions), open(f"data/model-debug/data_{dg.name}.pkl", "wb"))

        logger.info("Saving data complete.")
        return data
