import torch

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.simple import SimpleDatasetDefault
from src.modules.training.datasets.tensor_index import TensorIndex
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

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

        for dataset_group in data.grids:
            if dataset_group == DatasetGroup.ALL:
                continue
            targets = dataset_to_list(data, dataset_group)[1]
            target_ti = SimpleDatasetDefault.create_ti(self.info, discrete=False)
            preds = data.predictions[dataset_group]
            pred_ti = self.info.model_ti
            calc_accuracy(targets, target_ti, preds, pred_ti, "Validation")

        return data


def calc_accuracy(
    targets: list[list[torch.Tensor]],
    target_ti: TensorIndex,
    preds: list[torch.Tensor],
    pred_ti: TensorIndex,
    index_pretty_name: str,
):
    """Calculate the accuracy of the model.

    :param index_name: The name of the indice.
    :param index_pretty_name: The pretty name of the indice.
    """
    y_obs, y_reward = targets
    pred_obs, pred_reward = preds

    accuracy = torch.zeros(*y_obs.shape[:3], len(target_ti.observation))
    for obs_idx in range(len(pred_ti.observation)):
        y_obs_tmp = y_obs[..., target_ti.observation[obs_idx]].squeeze()
        pred_obs_tmp = torch.argmax(pred_obs[..., pred_ti.observation[obs_idx]], dim=3)
        accuracy[..., obs_idx] = (pred_obs_tmp == y_obs_tmp).float()

    # Check if the whole resulting observation is correct for each sample
    obs_whole_acc = accuracy.prod(dim=-1).prod(dim=-1).prod(dim=-1).mean()

    # Calculate the mean accuracy over all fields for each sample
    obs_field_acc = accuracy.prod(dim=-1).mean(dim=[0, 1, 2])

    # Check if the agent is predicted correctly
    obs_agent_pos = y_obs[..., target_ti.observation[3]].squeeze().nonzero()
    obs_agent_acc = accuracy[
        obs_agent_pos[:, 0], obs_agent_pos[:, 1], obs_agent_pos[:, 2], target_ti.observation[3]
    ].mean()

    # For all fields, where the agent shouldn't be, check if the agent is not predicted
    obs_non_agent_pos = torch.nonzero(y_obs[..., target_ti.observation[3]].squeeze() == 0)
    obs_non_agent_acc = accuracy[
        obs_non_agent_pos[:, 0], obs_non_agent_pos[:, 1], obs_non_agent_pos[:, 2], target_ti.observation[3]
    ].mean()

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
