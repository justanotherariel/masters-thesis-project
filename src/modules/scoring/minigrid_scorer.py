from typing import Any, List

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
            raw_data = dataset_to_list(data, DatasetGroup.TRAIN)
            raw_ti = TwoDDataset.create_ti(self.info)
            preds = data.predictions[DatasetGroup.TRAIN]
            model_ti = self.info.model_ti
            self.calc_accuracy(raw_data, raw_ti, preds, model_ti, "Train")

        if DatasetGroup.VALIDATION in data.predictions:
            raw_data = dataset_to_list(data, DatasetGroup.VALIDATION)
            raw_ti = TwoDDataset.create_ti(self.info)
            preds = data.predictions[DatasetGroup.VALIDATION]
            model_ti = self.info.model_ti
            self.calc_accuracy(raw_data, raw_ti, preds, model_ti, "Validation")

        return data

    def calc_accuracy(self, raw_data: List[List[torch.Tensor]], raw_ti: TokenIndex, preds: List[torch.Tensor], model_ti: TokenIndex, index_pretty_name: str):
        """Calculate the accuracy of the model.

        :param index_name: The name of the indice.
        :param index_pretty_name: The pretty name of the indice.
        """
        y_obs, y_reward = raw_data[1]
        pred_obs, pred_reward = preds
        
        accuracy = torch.zeros(*y_obs.shape[:3], len(raw_ti.observation))
        for obs_idx in range(len(model_ti.observation)):
            y_obs_tmp =  y_obs[..., raw_ti.observation[obs_idx]].squeeze()
            pred_obs_tmp = torch.argmax(pred_obs[..., model_ti.observation[obs_idx]], dim=3)
            accuracy[..., obs_idx] = (pred_obs_tmp == y_obs_tmp).float()

        # Check if the whole resulting observation is correct for each sample
        obs_whole_acc = accuracy.prod(dim=-1).prod(dim=-1).prod(dim=-1).mean()
        
        # Calculate the mean accuracy over all fields for each sample
        obs_field_acc = accuracy.prod(dim=-1).mean(dim=[0, 1, 2])
        
        # Check if the agent is predicted correctly
        obs_agent_pos = y_obs[..., raw_ti.observation[3]].squeeze().nonzero()
        obs_agent_acc = accuracy[obs_agent_pos[:, 0], obs_agent_pos[:, 1], obs_agent_pos[:, 2], raw_ti.observation[3]].mean()
        
        # For all fields, where the agent shouldn't be, check if the agent is not predicted
        obs_non_agent_pos = torch.nonzero(y_obs[..., raw_ti.observation[3]].squeeze() == 0)
        obs_non_agent_acc = accuracy[obs_non_agent_pos[:, 0], obs_non_agent_pos[:, 1], obs_non_agent_pos[:, 2], raw_ti.observation[3]].mean()

        # Check Reward accuracy
        reward_acc = torch.isclose(y_reward, pred_reward, atol=0.1).sum() / len(y_reward)
        
        # Average all accuracies
        acc = (obs_whole_acc + obs_field_acc + obs_agent_acc + obs_non_agent_acc + reward_acc) / 5

        # Log the results
        logger.info(f"{index_pretty_name}: Accuracy: {acc}")
        logger.info(f"{index_pretty_name}: Observation Whole Accuracy: {obs_whole_acc}")
        logger.info(f"{index_pretty_name}: Observation Field Accuracy: {obs_field_acc}")
        logger.info(f"{index_pretty_name}: Observation Agent Accuracy: {obs_agent_acc}")
        logger.info(f"{index_pretty_name}: Observation Non-Agent Accuracy: {obs_non_agent_acc}")
        logger.info(f"{index_pretty_name}: Reward Accuracy: {reward_acc}")

        logger.log_to_external(
            message={
                f"{index_pretty_name}/Accuracy": acc,
                f"{index_pretty_name}/Observation Whole Accuracy": obs_whole_acc,
                f"{index_pretty_name}/Observation Field Accuracy": obs_field_acc,
                f"{index_pretty_name}/Observation Agent Accuracy": obs_agent_acc,
                f"{index_pretty_name}/Observation Non-Agent Accuracy": obs_non_agent_acc,
                f"{index_pretty_name}/Reward Accuracy": reward_acc,
            },
        )
