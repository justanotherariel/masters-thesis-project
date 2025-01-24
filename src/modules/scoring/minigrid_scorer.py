from typing import Any

import torch

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import PipelineData, PipelineInfo, DatasetGroup
from src.modules.training.datasets.two_d_dataset import TwoDDataset
from .data_transform import dataset_to_list

logger = Logger()


class MinigridScorer(TransformationBlock):
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

        if DatasetGroup.TRAIN in data.predictions:
            targets = dataset_to_list(data, DatasetGroup.TRAIN)[1]
            self.calc_accuracy(data.predictions[DatasetGroup.TRAIN], targets, "Train")

        if DatasetGroup.VALIDATION in data.predictions:
            targets = dataset_to_list(data, DatasetGroup.VALIDATION)[1]
            self.calc_accuracy(data.predictions[DatasetGroup.VALIDATION], targets, "Validation")

        return data

    def calc_accuracy(self, predictions, targets, index_pretty_name: str):
        """Calculate the accuracy of the model.

        :param index_name: The name of the indice.
        :param index_pretty_name: The pretty name of the indice.
        """

        model_ti = self.info.model_ti
        # model_ti.discrete = True  # ToDo: fetch from info, supplied by model setup

        obs_pred, reward_pred = predictions
        obs_target, reward_target = targets

        # Check Observation object accuaracy
        obs_pred_obj = obs_pred[..., model_ti.observation[0]].argmax(axis=-1)
        obs_target_obj = obs_target[..., 0]
        obs_obj_accuracy = (obs_pred_obj == obs_target_obj).sum() / obs_pred_obj.numel()

        obs_obj_accuracy_no_walls = (obs_pred_obj[:, 1:-1, 1:-1] == obs_target_obj[:, 1:-1, 1:-1]).sum() / obs_pred_obj[
            :, 1:-1, 1:-1
        ].numel()

        # Check Observation color accuracy
        obs_pred_color = obs_pred[..., model_ti.observation[1]].argmax(axis=-1)
        obs_target_color = obs_target[..., 1]
        obs_color_accuracy = (obs_pred_color == obs_target_color).sum() / obs_pred_color.numel()

        # Check Observation state accuracy
        obs_pred_state = obs_pred[..., model_ti.observation[2]].argmax(axis=-1)
        obs_target_state = obs_target[..., 2]
        obs_state_accuracy = (obs_pred_state == obs_target_state).sum() / obs_pred_state.numel()

        # Check Observation agent accuracy
        obs_pred_agent = obs_pred[..., model_ti.observation[3]].argmax(axis=-1)
        obs_target_agent = obs_target[..., 3]
        obs_agent_accuracy = (obs_pred_agent == obs_target_agent).sum() / obs_pred_agent.numel()

        # Check Reward accuracy
        reward_accuracy = torch.isclose(reward_target, reward_pred, atol=0.1).sum() / len(reward_target)

        # Overall accuracy
        accuracy = (
            obs_obj_accuracy + obs_color_accuracy + obs_state_accuracy + obs_agent_accuracy + reward_accuracy
        ) / 5

        # Log the results
        logger.info(f"{index_pretty_name}: Accuracy: {accuracy}")
        logger.info(f"{index_pretty_name}: Observation Object Accuracy: {obs_obj_accuracy}")
        logger.info(f"{index_pretty_name}: Observation Object Accuracy (no walls): {obs_obj_accuracy_no_walls}")
        logger.info(f"{index_pretty_name}: Observation Color Accuracy: {obs_color_accuracy}")
        logger.info(f"{index_pretty_name}: Observation State Accuracy: {obs_state_accuracy}")
        logger.info(f"{index_pretty_name}: Observation Agent Accuracy: {obs_agent_accuracy}")
        logger.info(f"{index_pretty_name}: Reward Accuracy: {reward_accuracy}")

        logger.log_to_external(
            message={
                f"{index_pretty_name}/Accuracy": accuracy,
                f"{index_pretty_name}/Observation Object Accuracy": obs_obj_accuracy,
                f"{index_pretty_name}/Observation Object Accuracy (no walls)": obs_obj_accuracy_no_walls,
                f"{index_pretty_name}/Observation Color Accuracy": obs_color_accuracy,
                f"{index_pretty_name}/Observation State Accuracy": obs_state_accuracy,
                f"{index_pretty_name}/Observation Agent Accuracy": obs_agent_accuracy,
                f"{index_pretty_name}/Reward Accuracy": reward_accuracy,
            },
        )
