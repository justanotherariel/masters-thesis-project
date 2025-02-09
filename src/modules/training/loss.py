from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo


def ce_focal_loss(predictions, targets, weight=None, gamma=2.0):
    ce_loss = F.cross_entropy(predictions, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    return focal_loss


def ce_adaptive_loss(predictions, targets, beta=0.3):
    losses = F.cross_entropy(predictions, targets, reduction="none")
    k = int(beta * len(losses))
    _, indices = torch.topk(losses, k)
    return losses[indices]


def ce_rebalance_loss(predictions: torch.Tensor, targets: torch.Tensor):
    num_classes = predictions.size(1)
    class_counts = torch.bincount(targets, minlength=num_classes)

    weight = torch.where(class_counts > 0, torch.max(class_counts) / class_counts, torch.zeros_like(class_counts))

    return F.cross_entropy(predictions, targets, weight=weight, reduction="none")


def ce_rebalanced_focal_loss(predictions: torch.Tensor, targets: torch.Tensor, gamma=2.0):
    num_classes = predictions.size(1)
    class_counts = torch.bincount(targets, minlength=num_classes)

    weight = torch.where(
        class_counts > 0, torch.max(class_counts) / (class_counts + 1e-8), torch.zeros_like(class_counts)
    )

    return ce_focal_loss(predictions, targets, weight=weight, gamma=gamma, reduction="none")


class BaseLoss:
    def __init__(self, **kwargs):
        raise NotImplementedError("BaseLoss is an abstract class and should not be instantiated.")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")

    def __call__(
        self,
        predictions: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("BaseLoss is an abstract class and should not be called.")


@dataclass
class MinigridLoss(BaseLoss):
    discrete_loss_fn: callable = None
    obs_loss_weight: float = 0.7
    reward_loss_weight: float | None = None
    dynamic_var_weight: float = 1.0

    def __init__(self, **kwargs):
        self.discrete_loss_fn = kwargs.get("discrete_loss_fn")
        self.obs_loss_weight = kwargs.get("obs_loss_weight", 0.5)
        self.reward_loss_weight = kwargs.get("reward_loss_weight", 1 - self.obs_loss_weight)
        self.dynamic_var_weight = kwargs.get("dynamic_var_weight", 10.0)

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
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss for observation and reward predictions."""
        predicted_obs, predicted_reward = predictions
        target_obs, target_reward = targets
        feature_obs, feature_action = features

        # Compute observation loss
        # Assume that all observation variables are discrete
        obs_loss = torch.empty(*predicted_obs.shape[:3], len(self._tensor_values), device=predicted_obs.device)
        feature_obs_argmax = torch.empty(
            *feature_obs.shape[:3], len(self._tensor_values), dtype=torch.uint8, device=feature_obs.device
        )
        target_obs_argmax = torch.empty(
            *target_obs.shape[:3], len(self._tensor_values), dtype=torch.uint8, device=target_obs.device
        )
        for value_idx, value_range in enumerate(self._tensor_values):
            pred_range = predicted_obs[..., value_range]
            target_range = target_obs[..., value_range]
            loss = self.discrete_loss_fn(
                predictions=pred_range.reshape(-1, len(value_range)),
                targets=target_range.argmax(dim=-1).reshape(-1),
            )
            obs_loss[..., value_idx] = loss.reshape(obs_loss[..., value_idx].shape)

            feature_obs_argmax[..., value_idx] = feature_obs[..., value_range].argmax(dim=-1)
            target_obs_argmax[..., value_idx] = target_obs[..., value_range].argmax(dim=-1)
        obs_loss /= len(self._tensor_values)  # Normalize loss by number of observation variables

        # Incentivize learning fields and variables of that field that change instead of focusing on static fields
        changed_field_variables = feature_obs_argmax != target_obs_argmax
        # obs_loss[changed_field_variables] *= obs_loss.numel() / changed_field_variables.sum()
        # obs_loss[~changed_field_variables] *= changed_field_variables.sum() / obs_loss.numel()
        obs_loss[changed_field_variables] *= self.dynamic_var_weight
        obs_loss = obs_loss.mean()

        # Compute reward loss
        reward_loss = F.mse_loss(predicted_reward, target_reward)

        # Final loss combining original and consistency terms
        total_loss = self.obs_loss_weight * obs_loss + self.reward_loss_weight * reward_loss
        return total_loss
