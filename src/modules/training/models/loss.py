from torch.nn import functional as F
from torch import nn
import torch

def focal_loss(predictions, targets, gamma=2.0):
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def adaptive_loss(predictions, targets, beta=0.3):
    losses = F.cross_entropy(predictions, targets, reduction='none')
    k = int(beta * len(losses))
    _, indices = torch.topk(losses, k)
    return losses[indices].mean()

def rebalance_loss(predictions: torch.Tensor, targets: torch.Tensor):
    num_classes = predictions.size(1)
    # bincount requires CPU tensors to be long/int64 and GPU tensors to be int
    class_counts = torch.bincount(targets, minlength=num_classes)
    
    weights = torch.where(
        class_counts > 0,
        torch.max(class_counts) / (class_counts + 1e-8),
        torch.zeros_like(class_counts)
    )
    
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    sample_weights = weights[targets]
    return (ce_loss * sample_weights).mean()