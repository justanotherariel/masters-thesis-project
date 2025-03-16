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


def eta_focus_loss(eta, weight: float = 0.01):
    # Normalize each column (per output token) to create a probability distribution
    eta_norm = eta / (eta.sum(dim=1, keepdim=True) + 1e-10)

    # Calculate entropy for each output token's influence distribution
    # Lower entropy = more focused attention
    entropies = -torch.sum(eta_norm * torch.log(eta_norm + 1e-10), dim=1)

    # Return mean entropy as the loss
    return entropies.mean() * weight

def eta_minimization_loss(eta: torch.Tensor, loss_type: str = 'frobenius', weight: float = 0.01) -> torch.Tensor:
    """
    Computes a loss term that encourages minimizing eta values, effectively pushing
    attention toward the dummy token.
    
    Args:
        eta: Tensor of shape [batch_size, seq_length, seq_length] representing 
                  the final attention influence matrix (excluding dummy token)
        loss_type: The type of loss function to use ('l1', 'frobenius', 'nuclear', 'entropy', 'sparse')
        weight: Scaling factor to control the strength of the regularization
        
    Returns:
        A scalar loss value that can be added to the main loss function
    """
    # Ensure we have a proper batch dimension
    if eta.dim() == 2:
        eta = eta.unsqueeze(0)
    
    if loss_type == 'l1':
        # L1 loss: encourages sparsity by pushing most values toward zero
        return weight * torch.abs(eta).mean()
    
    elif loss_type == 'frobenius':
        # Frobenius norm: penalizes the squared magnitude of all attention values
        # This is like L2 regularization for matrices
        return weight * torch.sum(eta ** 2, dim=(1, 2)).mean()
    
    elif loss_type == 'nuclear':
        # Nuclear norm: sum of singular values, encourages low-rank attention patterns
        # This is computationally more expensive but can produce more structured attention
        loss = 0
        for i in range(eta.size(0)):  # For each item in batch
            # SVD to get singular values
            u, s, v = torch.svd(eta[i])
            # Sum of singular values = nuclear norm
            loss += torch.sum(s)
        return weight * loss / eta.size(0)
    
    elif loss_type == 'entropy':
        # Entropy loss: encourages attention to be concentrated rather than diffuse
        # First normalize eta to sum to 1
        eps = 1e-8  # Small epsilon to prevent log(0)
        eta_abs = torch.abs(eta)
        eta_sum = eta_abs.sum(dim=(1, 2), keepdim=True) + eps
        eta_prob = eta_abs / eta_sum
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(eta_prob * torch.log(eta_prob + eps), dim=(1, 2)).mean()
        return weight * entropy
    
    elif loss_type == 'sparse':
        # Sparsity-promoting loss based on L1/L2 ratio
        # Higher ratio = more sparsity
        l1_norm = torch.abs(eta).sum(dim=(1, 2))
        l2_norm = torch.sqrt((eta ** 2).sum(dim=(1, 2)) + 1e-8)
        sparsity_term = (l1_norm / l2_norm).mean()
        return weight * sparsity_term
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

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
        predictions: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, torch.Tensor],
        features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined loss for observation and reward predictions."""
        predicted_next_obs, predicted_reward = predictions[:2]
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
        if len(predictions) > 2 and self.eta_loss_fn is not None:
            eta_loss = self.eta_loss_fn(predictions[2])

        # Final loss combining original and consistency terms
        total_loss = self.obs_loss_weight * obs_loss.mean() + self.reward_loss_weight * reward_loss + eta_loss

        n_samples = predicted_next_obs.shape[0]

        # Logging
        losses = {
            "Loss": total_loss.expand(n_samples),
            "Observation Loss": (obs_loss.mean() * self.obs_loss_weight).expand(n_samples),
            "Observation Loss - Object": (
                obs_loss[0] * self.obs_loss_weight * (1 / len(self._tensor_values))
            ).expand(n_samples),
            "Observation Loss - Color": (
                obs_loss[1] * self.obs_loss_weight * (1 / len(self._tensor_values))
            ).expand(n_samples),
            "Observation Loss - State": (
                obs_loss[2] * self.obs_loss_weight * (1 / len(self._tensor_values))
            ).expand(n_samples),
            "Observation Loss - Agent": (
                obs_loss[3] * self.obs_loss_weight * (1 / len(self._tensor_values))
            ).expand(n_samples),
            "Reward Loss": (reward_loss * self.reward_loss_weight).expand(n_samples),
            "Eta Loss": eta_loss.expand(n_samples),
        }
        return total_loss, losses
