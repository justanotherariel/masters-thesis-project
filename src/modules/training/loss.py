from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo


def ce_focal_loss(predictions, targets, weight=None, gamma=2.0):
    ce_loss = F.cross_entropy(predictions, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss


def ce_adaptive_loss(predictions, targets, beta=0.3):
    losses = F.cross_entropy(predictions, targets, reduction="none")
    k = int(beta * len(losses))
    _, indices = torch.topk(losses, k)
    return losses[indices].mean()


def ce_rebalance_loss(predictions: torch.Tensor, targets: torch.Tensor):
    num_classes = predictions.size(1)
    class_counts = torch.bincount(targets, minlength=num_classes)

    weight = torch.where(
        class_counts > 0, torch.max(class_counts) / (class_counts + 1e-8), torch.zeros_like(class_counts)
    )

    return F.cross_entropy(predictions, targets, weight=weight)


def ce_rebalanced_focal_loss(predictions: torch.Tensor, targets: torch.Tensor, gamma=2.0):
    num_classes = predictions.size(1)
    class_counts = torch.bincount(targets, minlength=num_classes)

    weight = torch.where(
        class_counts > 0, torch.max(class_counts) / (class_counts + 1e-8), torch.zeros_like(class_counts)
    )

    return ce_focal_loss(predictions, targets, weight=weight, gamma=gamma)


class BaseLoss:
    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")


@dataclass
class MinigridLoss(BaseLoss):
    discrete_loss_fn: callable = None
    obs_loss_weight: float = 0.8
    reward_loss_weight: float = 0.2
    two_agent_penalty_loss_weight: float | None = None

    def __post_init__(self):
        if self.two_agent_penalty_loss_weight is None:
            self.two_agent_penalty_loss_weight = max(1 - self.obs_loss_weight - self.reward_loss_weight, 0)

        if self.discrete_loss_fn is None:
            raise ValueError("discrete_loss_fn must be provided")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        # Store value ranges for each grid cell component
        self._ti = info.model_ti
        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]
        return info

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined loss for observation and reward predictions."""
        predicted_next_obs, predicted_reward = predictions
        target_next_obs, target_reward = targets

        # Compute observation loss using cross entropy for value ranges (logits)
        obs_loss = torch.empty(len(self._tensor_values), device=predicted_next_obs.device)
        for value_idx, value_range in enumerate(self._tensor_values):
            # If it's a discrete value - more than one element, use cross entropy loss
            if len(value_range) > 1:
                pred_range = predicted_next_obs[..., value_range]
                target_range = target_next_obs[..., value_range]
                loss = self.discrete_loss_fn(
                    predictions=pred_range.reshape(-1, len(value_range)),
                    targets=target_range.argmax(dim=-1).reshape(-1),
                )
            else:  # One element means it's a continuous value
                # For continuous values, use MSE
                pred_range = predicted_next_obs[..., value_range]
                target_range = target_next_obs[..., value_range]
                loss = F.mse_loss(pred_range, target_range)
            obs_loss[value_idx] = loss

        # Compute Two Agent Penalty Loss
        two_agent_penalty_loss = torch.tensor(0.0, device=predicted_next_obs.device)
        if self.two_agent_penalty_loss_weight > 0:
            two_agent_penalty_loss = self.two_agent_penalty_loss(predicted_next_obs)

        # Compute reward loss
        reward_loss = F.mse_loss(predicted_reward, target_reward)

        # Final loss combining original and consistency terms
        total_loss = (
            self.obs_loss_weight * obs_loss.mean()
            + self.reward_loss_weight * reward_loss
            + self.two_agent_penalty_loss_weight * two_agent_penalty_loss
        )

        n_samples = predicted_next_obs.shape[0]

        # Logging
        losses = {
            "Loss": total_loss.expand(n_samples),
            "Observation Loss": (obs_loss.mean() * self.obs_loss_weight).expand(n_samples),
            "Observation Loss - Object": (obs_loss[0] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(
                n_samples),
            "Observation Loss - Color": (obs_loss[1] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(n_samples),
            "Observation Loss - State": (obs_loss[2] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(n_samples),
            "Observation Loss - Agent": (obs_loss[3] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(n_samples),
            "Two Agent Penalty": (two_agent_penalty_loss * self.two_agent_penalty_loss_weight).expand(n_samples),
            "Reward Loss": (reward_loss * self.reward_loss_weight).expand(n_samples),
        }
        return total_loss, losses

    def two_agent_penalty_loss(self, predicted_next_obs: torch.Tensor) -> torch.Tensor:
        # Get the model certainty of there being an agent in each grid cell
        agent_certainty = (1 - predicted_next_obs[..., self._ti.observation[3][0]]).view(
            predicted_next_obs.shape[0], -1
        )

        # Calculate certainty of having two agents in different locations
        joint_agent_certainty = torch.einsum("bi,bj->bij", agent_certainty, agent_certainty)

        # Zero out the main diagonal as we don't want to penalize the model for predicting an agent on the same location
        main_diag_mask = ~torch.eye(joint_agent_certainty.shape[1], dtype=bool, device=joint_agent_certainty.device)
        joint_agent_certainty = joint_agent_certainty * main_diag_mask

        # Calculate the penalty
        penalty = joint_agent_certainty.sum(dim=[1, 2]) / (joint_agent_certainty.shape[1] ** 2)

        return penalty.mean()
