from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.typing.pipeline_objects import PipelineInfo

EPS = 1e-8

@dataclass
class CELoss:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)

@dataclass
class CEFocalLoss:
    gamma: float = 2.0
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


@dataclass
class CEAdaptiveLoss:
    beta: float = 0.3
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(predictions, targets, reduction="none")
        k = int(self.beta * len(losses))
        _, indices = torch.topk(losses, k)
        return losses[indices].mean()


@dataclass
class CERebalanceLoss:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = predictions.size(1)
        class_counts = torch.bincount(targets, minlength=num_classes)

        weight = torch.where(
            class_counts > 0, torch.max(class_counts) / (class_counts + EPS), torch.zeros_like(class_counts)
        )

        return F.cross_entropy(predictions, targets, weight=weight)


@dataclass
class CERebalancedFocalLoss:
    gamma: float = 2.0
    
    def __post_init__(self):
        self.ce_focal_loss = CEFocalLoss(gamma=self.gamma)
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = predictions.size(1)
        class_counts = torch.bincount(targets, minlength=num_classes)

        weight = torch.where(
            class_counts > 0, torch.max(class_counts) / (class_counts + EPS), torch.zeros_like(class_counts)
        )

        return self.ce_focal_loss(predictions, targets, weight=weight)


@dataclass
class EtaL1Loss:
    weight: float = 0.01
    
    def __call__(self, eta: torch.Tensor) -> torch.Tensor:
        """
        L1 loss: encourages sparsity by pushing most values toward zero.
        All attention values are penalized equally.

        Lower values = more sparsity
        """
        return self.weight * torch.abs(eta).mean()


@dataclass
class EtaL2Loss:
    weight: float = 0.01
    
    def __call__(self, eta: torch.Tensor) -> torch.Tensor:
        """
        L2 loss / Frobenius norm: penalizes the squared magnitude of all attention values.
        Higher values are penalized more heavily than lower values.

        Lower values = more sparsity
        """
        return self.weight * torch.linalg.matrix_norm(eta, ord="fro", dim=(1, 2)).mean()


@dataclass
class EtaLRatioLoss:
    weight: float = 0.01
    
    def __call__(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Sparsity-promoting loss based on L1/L2 ratio.

        Lower ratio = more sparsity
        """
        l1_norm = torch.abs(eta).sum(dim=(1, 2))
        l2_norm = torch.linalg.matrix_norm(eta, ord="fro", dim=(1, 2))
        sparsity_term = (l1_norm / l2_norm).mean()
        return self.weight * sparsity_term


@dataclass
class EtaEntropyLoss:
    weight: float = 0.01
    
    def __call__(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Encourages a uniform distribution of attention weights
        Lower entropy = more focused attention
        """
        eta_prob = F.softmax(eta, dim=2) + EPS
        entropy = -(eta_prob * eta_prob.log()).sum(dim=2).mean()
        return self.weight * entropy


def eta_entropy_guided_l1_softplus_loss(
    eta: torch.Tensor, weight: float = 0.01, smoothness: float = 10.0
) -> torch.Tensor:
    """
    Computes a combined loss that uses entropy to guide L1 regularization:
    - When entropy is high (diffuse attention), L1 regularization is relaxed
    - When entropy is low (concentrated attention), L1 regularization is enforced

    Args:
        eta: Tensor of shape [batch_size, seq_length, seq_length] representing
             the final attention influence matrix (excluding dummy token)
        base_weight: Base scaling factor for the overall loss
        smoothness: Controls how quickly L1 weight transitions as entropy changes
                    Higher values = sharper transition

    Returns:
        A scalar loss value that can be added to the main loss function
    """
    # L1 component
    l1_norm = torch.abs(eta).mean()

    # Entropy component
    eta_prob = F.softmax(eta, dim=2) + EPS
    entropy = -(eta_prob * eta_prob.log()).sum(dim=2).mean(dim=1)

    # Normalize entropy to [0, 1] range
    max_entropy = torch.log(torch.tensor(eta.shape[1], dtype=torch.float, device=eta.device))
    normalized_entropy = entropy / max_entropy

    # Softplus smoothing for L1 coefficient
    # When entropy is high, coefficient is low; when entropy is low, coefficient is high
    l1_coef = torch.nn.functional.softplus(smoothness * (1.0 - normalized_entropy)) / smoothness

    # Final combined loss (mean over batch)
    return weight * (l1_coef * l1_norm).mean()


def eta_entropy_guided_l1_exp_loss(
    eta: torch.Tensor, base_weight: float = 0.01, decay_rate: float = 5.0
) -> torch.Tensor:
    # L1 component
    l1_norm = torch.abs(eta).mean()

    # Entropy component
    eta_prob = F.softmax(eta, dim=2) + EPS
    entropy = -(eta_prob * eta_prob.log()).sum(dim=2).mean(dim=1)

    # Normalize entropy to [0, 1] range
    max_entropy = torch.log(torch.tensor(eta.shape[1], dtype=torch.float, device=eta.device))
    normalized_entropy = entropy / max_entropy

    # Exponential scaling: when entropy is high, coefficient is close to 0
    l1_coef = torch.exp(-decay_rate * normalized_entropy)

    # Final combined loss (mean over batch)
    return base_weight * (l1_coef * l1_norm).mean()


def eta_entropy_guided_l1_sigmoid_loss(
    eta: torch.Tensor, weight: float = 0.01, scale: float = 5.0, midpoint: float = 0.5
) -> torch.Tensor:
    """
    Combines L1 and entropy losses with adaptive sigmoid weighting.
    When entropy is high (unfocused attention), L1 weight is reduced.
    When entropy is low (focused attention), L1 weight is increased.

    Args:
        eta: Attention influence matrix
        weight: Overall weight of the combined loss
        scale: Controls steepness of sigmoid transition (higher = sharper transition)
        midpoint: Normalized entropy value where L1 weight equals 0.5
    """
    # L1 component
    l1_norm = torch.abs(eta).mean()

    # Entropy component
    eta_prob = F.softmax(eta, dim=2) + EPS
    entropy = -(eta_prob * eta_prob.log()).sum(dim=2).mean(dim=1)

    # Normalize entropy to [0, 1] range
    max_entropy = torch.log(torch.tensor(eta.shape[1], dtype=torch.float, device=eta.device))
    normalized_entropy = entropy / max_entropy

    # Sigmoid weighting: smooth transition from low to high L1 weight as entropy decreases
    adaptive_weight = torch.sigmoid(scale * (midpoint - normalized_entropy))

    return weight * (adaptive_weight * l1_norm).mean()


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
    eta_loss_fn: callable = None

    obs_loss_weight: float = 0.8
    reward_loss_weight: float = 0.2

    def __post_init__(self):
        if self.discrete_loss_fn is None:
            raise ValueError("discrete_loss_fn must be provided")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        # Store value ranges for each grid cell component
        self._ti = info.model_ti
        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]
        return info

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined loss for observation and reward predictions."""
        predicted_next_obs, predicted_reward = predictions["pred_obs"], predictions["pred_reward"]
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

        # Compute reward loss
        reward_loss = F.mse_loss(predicted_reward, target_reward)

        # Compute auxiliary losses
        eta_loss = torch.tensor(0.0, device=predicted_next_obs.device)
        if "attention_sum" in predictions and self.eta_loss_fn is not None:
            eta_loss = self.eta_loss_fn(predictions["attention_sum"])

        # Final loss combining original and consistency terms
        total_loss = self.obs_loss_weight * obs_loss.mean() + self.reward_loss_weight * reward_loss + eta_loss

        n_samples = predicted_next_obs.shape[0]

        # Logging
        losses = {
            "Loss": total_loss.expand(n_samples),
            "Observation Loss": (obs_loss.mean() * self.obs_loss_weight).expand(n_samples),
            "Observation Loss - Object": (obs_loss[0] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(
                n_samples
            ),
            "Observation Loss - Color": (obs_loss[1] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(
                n_samples
            ),
            "Observation Loss - State": (obs_loss[2] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(
                n_samples
            ),
            "Observation Loss - Agent": (obs_loss[3] * self.obs_loss_weight * (1 / len(self._tensor_values))).expand(
                n_samples
            ),
            "Reward Loss": (reward_loss * self.reward_loss_weight).expand(n_samples),
            "Eta Loss": eta_loss.expand(n_samples),
        }
        return total_loss, losses
