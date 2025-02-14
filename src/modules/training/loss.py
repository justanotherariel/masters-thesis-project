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
    def __init__(self, **kwargs):
        raise NotImplementedError("BaseLoss is an abstract class and should not be instantiated.")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ):
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")


@dataclass
class MinigridLoss(BaseLoss):
    discrete_loss_fn: callable = None
    obs_loss_weight: float = 0.7
    reward_loss_weight: float = 0.2

    def __init__(self, **kwargs):
        self.discrete_loss_fn = kwargs.get("discrete_loss_fn")
        self.obs_loss_weight = kwargs.get("obs_loss_weight", 0.5)
        self.reward_loss_weight = kwargs.get("reward_loss_weight", 0.5)

        if self.discrete_loss_fn is None:
            raise ValueError("discrete_loss_fn must be provided")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        # Store softmax ranges for each grid cell component
        self._ti = info.model_ti
        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]
        return info

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Compute the combined loss for observation and reward predictions."""
        predicted_next_obs, predicted_reward = predictions
        target_next_obs, target_reward = targets

        # Compute observation loss using cross entropy for softmaxed ranges
        obs_loss = 0
        for value_range in self._tensor_values:
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
            obs_loss += loss / len(self._tensor_values)

        # Compute reward loss
        reward_loss = F.mse_loss(predicted_reward, target_reward)

        # Final loss combining original and consistency terms
        total_loss = self.obs_loss_weight * obs_loss + self.reward_loss_weight * reward_loss
        return total_loss
