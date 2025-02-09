from dataclasses import dataclass

import torch

from src.typing.pipeline_objects import PipelineInfo


class BaseAccuracy:
    def __init__(self, **kwargs):
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be instantiated.")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be called.")

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
    ) -> dict[str, float]:
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be called.")


@dataclass
class MinigridAccuracy(BaseAccuracy):
    def __init__(self, **kwargs):
        pass

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._into = info
        self._ti = info.model_ti

        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
    ) -> dict[str, float]:
        pred_obs, pred_reward = predictions
        target_obs, target_reward = targets
        feature_obs, feature_action = features

        pred_obs_argmax = torch.empty(*target_obs.shape[:3], len(self._ti.observation), dtype=torch.uint8)
        target_obs_argmax = torch.empty(*target_obs.shape[:3], len(self._ti.observation), dtype=torch.uint8)
        feature_obs_argmax = torch.empty(*feature_obs.shape[:3], len(self._ti.observation), dtype=torch.uint8)
        for idx, values in enumerate(self._tensor_values):
            pred_obs_argmax[..., idx] = pred_obs[..., values].argmax(dim=-1)
            target_obs_argmax[..., idx] = target_obs[..., values].argmax(dim=-1)
            feature_obs_argmax[..., idx] = feature_obs[..., values].argmax(dim=-1)

        object_acc = self._calc_object_color_acc(pred_obs_argmax, target_obs_argmax)
        color_acc = self._calc_color_acc(pred_obs_argmax, target_obs_argmax)
        state_acc = self._calc_state_acc(pred_obs_argmax, target_obs_argmax)

        agent_acc = self._calc_agent_acc(pred_obs_argmax, target_obs_argmax, feature_obs_argmax)

        accuracies = {
            "Object Accuracy": object_acc,
            "Color Accuracy": color_acc,
            "State Accuracy": state_acc,
        }
        accuracies.update(agent_acc)
        return accuracies

    def _calc_object_color_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate the accuracy of the object component of the observation tensor. Each object class is weighted
        equally.
        """

        target_class_counts = torch.bincount(targets[..., 0].view(-1), minlength=self._ti.info["observation"][0][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = targets[..., 0] == predictions[..., 0]
        pred_class_counts = torch.bincount(targets[..., 0][truth_mask], minlength=self._ti.info["observation"][0][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]

        return (pred_class_counts / target_class_counts).mean().item()

    def _calc_color_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate the accuracy of the color component of the observation tensor. Each state class is weighted
        equally, but is only counted if the object component on that field was correctly predicted.
        """
        org_target_class_counts = torch.bincount(targets[..., 1].view(-1), minlength=self._ti.info["observation"][1][1])
        zero_class_counts = org_target_class_counts == 0
        org_target_class_counts = org_target_class_counts[~zero_class_counts]

        object_truth_mask = targets[..., 0] == predictions[..., 0]
        predictions = predictions[object_truth_mask]
        targets = targets[object_truth_mask]

        target_class_counts = torch.bincount(targets[..., 1].view(-1), minlength=self._ti.info["observation"][1][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = targets[..., 1] == predictions[..., 1]
        pred_class_counts = torch.bincount(targets[..., 1][truth_mask], minlength=self._ti.info["observation"][1][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]

        return (pred_class_counts.sum() / org_target_class_counts.sum()).item()

    def _calc_state_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate the accuracy of the state component of the observation tensor. Each state class is weighted
        equally, but is only counted if the object component on that field was correctly predicted.
        """
        org_target_class_counts = torch.bincount(targets[..., 2].view(-1), minlength=self._ti.info["observation"][2][1])
        zero_class_counts = org_target_class_counts == 0
        org_target_class_counts = org_target_class_counts[~zero_class_counts]

        object_truth_mask = targets[..., 0] == predictions[..., 0]
        predictions = predictions[object_truth_mask]
        targets = targets[object_truth_mask]

        target_class_counts = torch.bincount(targets[..., 2].view(-1), minlength=self._ti.info["observation"][2][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = targets[..., 2] == predictions[..., 2]
        pred_class_counts = torch.bincount(targets[..., 2][truth_mask], minlength=self._ti.info["observation"][2][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]

        return (pred_class_counts.sum() / org_target_class_counts.sum()).item()

    def _calc_agent_acc(self, predictions: torch.Tensor, targets: torch.Tensor, features: torch.Tensor):
        # How often was only one agent in the observation tensor?
        samples_one_agent_pred = (predictions[..., 3].view(-1, predictions.shape[1] * predictions.shape[2]) != 0).sum(
            dim=1
        ) == 1
        perc_samples_one_agent = samples_one_agent_pred.sum().item() / targets.shape[0]

        def agent_correct_perc(pred, targ, mask, max_correct_samples):
            predictions_tmp = pred[mask, :, :, 3]
            targets_tmp = targ[mask, :, :, 3]
            agent_location = targets_tmp != 0
            correct = predictions_tmp[agent_location] == targets_tmp[agent_location]
            return correct.sum().item() / max_correct_samples

        # For the samples with only one agent, how often was the agent correctly predicted?
        perc_samples_agent_correct = agent_correct_perc(predictions, targets, samples_one_agent_pred, targets.shape[0])

        # Find samples where the agent was supposed to stay
        samples_stay_mask = (features[..., 3] == targets[..., 3]).all(dim=[1, 2])
        perc_samples_agent_stay_correct = agent_correct_perc(
            predictions, targets, samples_stay_mask & samples_one_agent_pred, samples_stay_mask.sum().item()
        )

        # Find samples where the agent was supposed to rotate
        samples_same_field_mask = ((features[..., 3] != 0) == (targets[..., 3] != 0)).all(dim=[1, 2]) & ~samples_stay_mask
        perc_samples_agent_rotated_correct = agent_correct_perc(
            predictions, targets, samples_same_field_mask & samples_one_agent_pred, samples_same_field_mask.sum().item()
        )

        # Find samples where the agent was supposed to move
        samples_moved_mask = (((features[..., 3] != 0) != (targets[..., 3] != 0)).sum(dim=[1, 2]) == 2) & ~samples_stay_mask & ~samples_same_field_mask
        perc_samples_agent_moved_correct = agent_correct_perc(
            predictions, targets, samples_moved_mask & samples_one_agent_pred, samples_moved_mask.sum().item()
        )

        return {
            "One Agent Samples Accuracy": perc_samples_one_agent,
            "Agent Accuracy": perc_samples_agent_correct,
            "Agent Stay Accuracy": perc_samples_agent_stay_correct,
            "Agent Rotate Accuracy": perc_samples_agent_rotated_correct,
            "Agent Move Accuracy": perc_samples_agent_moved_correct,
        }
