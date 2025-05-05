from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo

EPS = 1e-8


class BaseAccuracy:
    def __init__(self, **kwargs):
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be instantiated.")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be called.")

    def __call__(
        self,
        predictions: tuple[torch.Tensor, ...],
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
        predictions: dict[str, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
    ) -> dict[str, float]:
        pred_obs, pred_reward = predictions["pred_obs"], predictions["pred_reward"]
        target_obs, target_reward = targets
        feature_obs, feature_action = features

        pred_obs_argmax = obs_argmax(pred_obs, self._ti, constrain_to_one_agent=self.constrain_to_one_agent)
        target_obs_argmax = obs_argmax(target_obs, self._ti)
        feature_obs_argmax = obs_argmax(feature_obs, self._ti)

        accuracies = {}
        accuracies.update(self._calc_transition_acc(pred_obs_argmax, pred_reward, target_obs_argmax, target_reward, feature_action))
        accuracies.update(self._calc_cell_acc(pred_obs_argmax, target_obs_argmax))
        accuracies.update(self._calc_agent_acc(pred_obs_argmax, target_obs_argmax, feature_obs_argmax, feature_action))
        accuracies.update(self._calc_reward_acc(pred_reward, target_reward))

        accuracies.update(self._calc_eta_metrics(predictions))

        return accuracies

    def _calc_transition_acc(
        self,
        pred_obs_argmax: torch.Tensor,
        pred_reward: torch.Tensor,
        target_obs_argmax: torch.Tensor,
        target_reward: torch.Tensor,
        feature_action: torch.Tensor,
    ):
        observation_correct = (pred_obs_argmax == target_obs_argmax).all(dim=[1, 2, 3])
        reward_correct = torch.isclose(pred_reward, target_reward, atol=0.2).squeeze()
        total_correct = observation_correct & reward_correct
        
        mask_forward = (feature_action[..., 2] == 1)
        samples_forward = total_correct[mask_forward]
        
        mask_rotate = (feature_action[..., 0] == 1) | (feature_action[..., 1] == 1)
        samples_rotate = total_correct[mask_rotate]

        return {
            "Transition Accuracy": total_correct,
            "Transition Accuracy Forward": samples_forward,
            "Transition Accuracy Rotate": samples_rotate,
        }

    def _calc_cell_acc(self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor):
        """
        Calculate the accuracy of the object component of the observation tensor. Each object class is weighted
        equally.
        """
        obj_correct = (pred_obs_argmax[..., 0] == target_obs_argmax[..., 0]).all(dim=[1, 2])
        color_correct = (pred_obs_argmax[..., 1] == target_obs_argmax[..., 1]).all(dim=[1, 2])
        state_correct = (pred_obs_argmax[..., 2] == target_obs_argmax[..., 2]).all(dim=[1, 2])
        cell_correct = (obj_correct & color_correct & state_correct).float()

        return {
            "Cell Accuracy": cell_correct,
            "Object Accuracy": obj_correct.float(),
            "Color Accuracy": color_correct.float(),
            "State Accuracy": state_correct.float(),
        }

    def _calc_agent_acc(
        self, pred_obs_argmax: torch.Tensor, target_obs_argmax: torch.Tensor, feature_obs_argmax: torch.Tensor, feature_action: torch.Tensor
    ):
        # How often was only one agent in the observation tensor?
        samples_one_agent = (
            pred_obs_argmax[..., 3].view(-1, pred_obs_argmax.shape[1] * pred_obs_argmax.shape[2]) != 0
        ).sum(dim=1) == 1

        # How often was the agent correctly predicted?
        samples_agent_correct = (pred_obs_argmax[..., 3] == target_obs_argmax[..., 3]).all(dim=[1, 2])
        
        # Action = left, right
        mask_agent_rotate = (feature_action[..., 0] == 1) | (feature_action[..., 1] == 1)
        samples_agent_rotate = (
            pred_obs_argmax[mask_agent_rotate, :, :, 3] == target_obs_argmax[mask_agent_rotate, :, :, 3]
        ).all(dim=[1, 2])
        
        # Action = forward
        mask_agent_forward = (feature_action[..., 2] == 1)
        samples_agent_forward = (
            pred_obs_argmax[mask_agent_forward, :, :, 3] == target_obs_argmax[mask_agent_forward, :, :, 3]
        ).all(dim=[1, 2])
        
        # Find samples where the agent was supposed to stay
        mask_agent_stay = mask_agent_forward & (feature_obs_argmax[..., 3] == target_obs_argmax[..., 3]).all(dim=[1, 2])
        samples_agent_stay = (
            pred_obs_argmax[mask_agent_stay, :, :, 3] == target_obs_argmax[mask_agent_stay, :, :, 3]
        ).all(dim=[1, 2])

        # Find samples where the agent was supposed to move
        mask_agent_forward = mask_agent_forward & (~mask_agent_stay & ~mask_agent_rotate)
        samples_agent_move = (
            pred_obs_argmax[mask_agent_forward, :, :, 3] == target_obs_argmax[mask_agent_forward, :, :, 3]
        ).all(dim=[1, 2])

        return {
            "One Agent Samples Accuracy": samples_one_agent,
            "Agent Accuracy": samples_agent_correct,
            "Agent Rotate Accuracy": samples_agent_rotate,
            "Agent Forward Accuracy": samples_agent_forward,
            "Agent Forward Stay Accuracy": samples_agent_stay,
            "Agent Forward Move Accuracy": samples_agent_move,
        }
        
    def _calc_reward_acc(self, pred_reward: torch.Tensor, target_reward: torch.Tensor):
        reward_correct = torch.isclose(pred_reward, target_reward, atol=0.2)

        reward_mask = target_reward != 0
        reward_pos_correct = torch.isclose(pred_reward[reward_mask], target_reward[reward_mask], atol=0.2)
        reward_neg_correct = torch.isclose(pred_reward[~reward_mask], target_reward[~reward_mask], atol=0.2)

        return {
            "Reward Accuracy": reward_correct,
            "Reward-Pos Accuracy": reward_pos_correct,
            "Reward-Neg Accuracy": reward_neg_correct,
        }

    def _calc_eta_metrics(self, predictions: dict[str, torch.Tensor]):
        results = {}
        
        attention_sum = predictions.get("attention_sum", None)
        eta = predictions.get("eta", None)
        
        if attention_sum is not None:
            results.update({"Eta Prob Sum": torch.abs(attention_sum).sum(dim=[1, 2])})

        if eta is not None:
            results.update({"Eta Sum": torch.abs(eta).sum(dim=[1, 2])})
            results.update({"Eta Mean": torch.abs(eta).mean(dim=[1, 2])})

        return results


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
