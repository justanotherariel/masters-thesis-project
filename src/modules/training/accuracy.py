from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

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
    constrain_to_one_agent: bool = field(default=False, repr=False)

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

        pred_obs_argmax = obs_argmax(pred_obs, self._ti, constrain_to_one_agent=self.constrain_to_one_agent)
        target_obs_argmax = obs_argmax(target_obs, self._ti)
        feature_obs_argmax = obs_argmax(feature_obs, self._ti)

        accuracies = {}
        accuracies.update(self._calc_transition_acc(pred_obs_argmax, pred_reward, target_obs_argmax, target_reward))
        
        accuracies.update(self._calc_object_acc(pred_obs_argmax, target_obs_argmax))
        accuracies.update(self._calc_color_acc(pred_obs_argmax, target_obs_argmax))
        accuracies.update(self._calc_state_acc(pred_obs_argmax, target_obs_argmax))
        accuracies.update(self._calc_agent_acc(pred_obs_argmax, target_obs_argmax, feature_obs_argmax))
        accuracies.update(self._calc_reward_acc(pred_reward, target_reward))

        return accuracies

    def _calc_transition_acc(
        self,
        pred_obs_argmax: torch.Tensor,
        pred_reward: torch.Tensor,
        target_obs_argmax: torch.Tensor,
        target_reward: torch.Tensor,
    ):
        observation_correct = (pred_obs_argmax == target_obs_argmax).all(dim=[1, 2, 3])
        reward_correct = torch.isclose(pred_reward, target_reward, atol=0.1).squeeze()
        total_correct = (observation_correct & reward_correct).sum().item()
        n_samples = target_obs_argmax.shape[0]

        return {
            "Transition Accuracy": total_correct / n_samples,
        }

    def _calc_object_acc(self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor):
        """
        Calculate the accuracy of the object component of the observation tensor. Each object class is weighted
        equally.
        """

        target_class_counts = torch.bincount(
            target_obs_argmax[..., 0].view(-1), minlength=self._ti.info["observation"][0][1]
        )
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = target_obs_argmax[..., 0] == pred_obs_argmax[..., 0]
        pred_class_counts = torch.bincount(
            target_obs_argmax[..., 0][truth_mask], minlength=self._ti.info["observation"][0][1]
        )
        pred_class_counts = pred_class_counts[~zero_class_counts]
        
        acc = (pred_class_counts / target_class_counts).mean().item()
        return { "Object Accuracy": acc }

    def _calc_color_acc(self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor):
        """
        Calculate the accuracy of the color component of the observation tensor. Each state class is weighted
        equally, but is only counted if the object component on that field was correctly predicted.
        """
        org_target_class_counts = torch.bincount(
            target_obs_argmax[..., 1].view(-1), minlength=self._ti.info["observation"][1][1]
        )
        zero_class_counts = org_target_class_counts == 0
        org_target_class_counts = org_target_class_counts[~zero_class_counts]

        object_truth_mask = target_obs_argmax[..., 0] == pred_obs_argmax[..., 0]
        pred_obs_argmax = pred_obs_argmax[object_truth_mask]
        target_obs_argmax = target_obs_argmax[object_truth_mask]

        target_class_counts = torch.bincount(
            target_obs_argmax[..., 1].view(-1), minlength=self._ti.info["observation"][1][1]
        )
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = target_obs_argmax[..., 1] == pred_obs_argmax[..., 1]
        pred_class_counts = torch.bincount(
            target_obs_argmax[..., 1][truth_mask], minlength=self._ti.info["observation"][1][1]
        )
        pred_class_counts = pred_class_counts[~zero_class_counts]

        acc = (pred_class_counts.sum() / org_target_class_counts.sum()).item()
        return { "Color Accuracy": acc }

    def _calc_state_acc(self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor):
        """
        Calculate the accuracy of the state component of the observation tensor. Each state class is weighted
        equally, but is only counted if the object component on that field was correctly predicted.
        """
        org_target_class_counts = torch.bincount(
            target_obs_argmax[..., 2].view(-1), minlength=self._ti.info["observation"][2][1]
        )
        zero_class_counts = org_target_class_counts == 0
        org_target_class_counts = org_target_class_counts[~zero_class_counts]

        object_truth_mask = target_obs_argmax[..., 0] == pred_obs_argmax[..., 0]
        pred_obs_argmax = pred_obs_argmax[object_truth_mask]
        target_obs_argmax = target_obs_argmax[object_truth_mask]

        target_class_counts = torch.bincount(
            target_obs_argmax[..., 2].view(-1), minlength=self._ti.info["observation"][2][1]
        )
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = target_obs_argmax[..., 2] == pred_obs_argmax[..., 2]
        pred_class_counts = torch.bincount(
            target_obs_argmax[..., 2][truth_mask], minlength=self._ti.info["observation"][2][1]
        )
        pred_class_counts = pred_class_counts[~zero_class_counts]

        acc = (pred_class_counts.sum() / org_target_class_counts.sum()).item()
        return { "State Accuracy": acc }

    def _calc_agent_acc(
        self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor, feature_obs_argmax: torch.Tensor
    ):
        # How often was only one agent in the observation tensor?
        samples_one_agent_pred = (
            pred_obs_argmax[..., 3].view(-1, pred_obs_argmax.shape[1] * pred_obs_argmax.shape[2]) != 0
        ).sum(dim=1) == 1
        perc_samples_one_agent = samples_one_agent_pred.sum().item() / target_obs_argmax.shape[0]

        def agent_correct_perc(pred, targ, mask, max_correct_samples):
            predictions_tmp = pred[mask, :, :, 3]
            targets_tmp = targ[mask, :, :, 3]
            agent_location = targets_tmp != 0
            correct = predictions_tmp[agent_location] == targets_tmp[agent_location]
            return correct.sum().item() / max_correct_samples

        # For the samples with only one agent, how often was the agent correctly predicted?
        perc_samples_agent_correct = agent_correct_perc(
            pred_obs_argmax, target_obs_argmax, samples_one_agent_pred, target_obs_argmax.shape[0]
        )

        # Find samples where the agent was supposed to stay
        samples_stay_mask = (feature_obs_argmax[..., 3] == target_obs_argmax[..., 3]).all(dim=[1, 2])
        perc_samples_agent_stay_correct = agent_correct_perc(
            pred_obs_argmax,
            target_obs_argmax,
            samples_stay_mask & samples_one_agent_pred,
            samples_stay_mask.sum().item(),
        )

        # Find samples where the agent was supposed to rotate
        samples_same_field_mask = ((feature_obs_argmax[..., 3] != 0) == (target_obs_argmax[..., 3] != 0)).all(
            dim=[1, 2]
        ) & ~samples_stay_mask
        perc_samples_agent_rotated_correct = agent_correct_perc(
            pred_obs_argmax,
            target_obs_argmax,
            samples_same_field_mask & samples_one_agent_pred,
            samples_same_field_mask.sum().item(),
        )

        # Find samples where the agent was supposed to move
        samples_moved_mask = (
            (((feature_obs_argmax[..., 3] != 0) != (target_obs_argmax[..., 3] != 0)).sum(dim=[1, 2]) == 2)
            & ~samples_stay_mask
            & ~samples_same_field_mask
        )
        perc_samples_agent_moved_correct = agent_correct_perc(
            pred_obs_argmax,
            target_obs_argmax,
            samples_moved_mask & samples_one_agent_pred,
            samples_moved_mask.sum().item(),
        )

        return {
            "One Agent Samples Accuracy": perc_samples_one_agent,
            "Agent Accuracy": perc_samples_agent_correct,
            "Agent Stay Accuracy": perc_samples_agent_stay_correct,
            "Agent Rotate Accuracy": perc_samples_agent_rotated_correct,
            "Agent Move Accuracy": perc_samples_agent_moved_correct,
        }

    def _calc_reward_acc(self, pred_reward: torch.Tensor, target_reward: torch.Tensor):
        reward_mask = target_reward != 0

        reward_predicted_correct = torch.isclose(pred_reward[reward_mask], target_reward[reward_mask], atol=0.1).sum().item()

        no_reward_predicted_correct = torch.isclose(pred_reward[~reward_mask], target_reward[~reward_mask], atol=0.1).sum().item()

        return {
            "Reward Accuracy": reward_predicted_correct / reward_mask.sum().item(),
            "No Reward Accuracy": no_reward_predicted_correct / (~reward_mask).sum().item(),
        }


def obs_argmax(obs, ti, *, constrain_to_one_agent: bool = False):
    tensor_values = [ti.observation[i] for i in range(len(ti.observation))]
    tensor_values = tensor_values[:-1] if constrain_to_one_agent else tensor_values

    obs_argmax = torch.zeros(*obs.shape[:3], len(ti.observation), dtype=torch.uint8, device=obs.device)
    for idx, values in enumerate(tensor_values):
        obs_argmax[..., idx] = obs[..., values].argmax(dim=-1)

    if constrain_to_one_agent:
        # Softmax the logits and find the most probable agent location
        agent_softmax = F.softmax(obs[..., ti.observation[3]], dim=-1)
        min_values, min_indices = agent_softmax[..., 0].view(obs.shape[0], -1).min(dim=1)

        # Find the agent in the observation tensor
        x_coords = min_indices // obs.shape[2]
        y_coords = min_indices % obs.shape[2]

        batch_indices = torch.arange(obs.shape[0])
        agent_values = obs[batch_indices, x_coords, y_coords][..., ti.observation[3]].argmax(dim=-1).to(torch.uint8)

        obs_argmax[batch_indices, x_coords, y_coords, 3] = agent_values

    return obs_argmax
