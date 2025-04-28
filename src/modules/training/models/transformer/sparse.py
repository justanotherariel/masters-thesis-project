"""Transformer model adapted to RL transition models.

Heavily inspired by
    Author : Hyunwoong
    Github : https://github.com/hyunwoongko/transformer
"""

import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention as described in 'Attention Is All You Need' paper.

    The attention mechanism computes the relevance between queries and keys, then uses
    these attention weights to create a weighted sum of the values.

    Formula: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    """

    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes attention scores and applies them to values.

        Args:
            query: Tensor of shape [batch_size, n_heads, seq_length, d_k]
            key: Tensor of shape [batch_size, n_heads, seq_length, d_k]
            value: Tensor of shape [batch_size, n_heads, seq_length, d_v]
            attention_mask: Optional boolean mask of shape [batch_size, n_heads, seq_length, seq_length]

        Returns:
            tuple: (weighted_values, attention_scores)

        Raises:
            RuntimeError: If input tensor dimensions don't match expected shapes
        """

        batch_size, n_heads, seq_length, dim_per_head = key.size()
        
        # 1. Compute attention scores
        key_transpose = key.transpose(2, 3)  # transpose
        attention_scores = (query @ key_transpose) / math.sqrt(dim_per_head)

        # 2. Convert scores to probabilities
        attention_probs = self.softmax(attention_scores)
        
        # 3. Mask out low attention scores
        low_attention_mask = (attention_probs < 0.1)
        attention_probs_model = attention_probs.masked_fill(low_attention_mask, 0.0)

        # 4. Apply attention dropout
        attention_probs_model = self.dropout(attention_probs_model)

        # 5. Compute weighted values
        weighted_values = attention_probs_model @ value
        
        return weighted_values, attention_probs, (~low_attention_mask).float()


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop_p: float = 0.1, use_bias: bool = True, idx: int = 0):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.idx = idx

        self.attention = ScaledDotProductAttention(attention_dropout=drop_p)

        # Linear projections
        self.query_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Applies multi-headed attention to the input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
            prev_eta: Optional previous eta values
            attention_mask: Optional attention mask

        Returns:
            tuple: (output_tensor, new_eta)
        """
        # 1. Project input into query, key, and value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # 2. Split projections into multiple heads
        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(value)

        # 3. Apply scaled dot-product attention
        attention_output, attention_probs, attention_masks = self.attention(
            query_heads, key_heads, value_heads
        )

        # 4. Merge attention heads and project to output dimension
        merged_attention = self.merge_heads(attention_output)
        output = self.output_projection(merged_attention)
        
        # 5. Compute attention map: sum of attention probabilities across heads
        attention_mask = attention_masks.sum(dim=1)
        attention_sum = attention_probs.sum(dim=1)
        
        return output, attention_sum, attention_mask

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the last dimension of the input tensor into n_heads."""
        batch_size, seq_length, d_model = x.size()

        x = x.view(batch_size, seq_length, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merges the attention heads back into a single tensor."""
        batch_size, n_heads, seq_length, d_k = x.size()

        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, self.d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob, idx: int = 0):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=d_model, n_heads=n_heads, drop_p=0.0, idx=idx)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Performs a single transformer layer operation.

        Args:
            x (torch.Tensor): Input Tensor
            prev_eta (torch.Tensor | None, optional): Previous Eta. Defaults to None.
            attention_mask (torch.Tensor | None, optional): Attention Mask. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: output tensor and new eta
        """

        # Self attention
        delta_x, attention_sum, attention_mask = self.attention(x)
        delta_x = self.dropout1(delta_x)
        x = self.norm1(x + delta_x)
        
        # Feed forward
        delta_x = self.ffn(x)
        delta_x = self.dropout2(delta_x)
        x = self.norm2(x + delta_x)

        return x, attention_sum, attention_mask


class SepAction(nn.Module):
    def __init__(
        self,
        in_obs_shape: tuple[int, int, int],
        out_obs_shape: tuple[int, int, int],
        action_dim: int,
        d_model: int,  # Internal representation dimension
        n_heads: int,  # Number of attention heads
        n_layers: int,  # Number of transformer layers
        d_ff: int,  # Feed-forward network hidden dimension
        drop_prob: float = 0.1,  # Dropout probability
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]
        self.input_token_len = self.obs_token_len + 1 + 1  # obs + action + reward

        # Input Projection
        self.obs_in_proj = nn.Linear(in_obs_shape[-1], d_model)
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.reward_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_token_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob, idx=idx)
                for idx in range(n_layers)
            ]
        )

        # Output projection
        self.obs_out_proj = nn.Linear(d_model, out_obs_shape[-1])
        self.reward_out_proj = nn.Linear(d_model, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack input
        x_obs = x[0].float()
        x_action = x[1].float()

        # Vars
        n_samples = x_obs.shape[0]

        # Embed input sequence
        x_obs = x_obs.view(n_samples, self.obs_token_len, self.in_obs_shape[-1])
        x_obs = self.obs_in_proj(x_obs)
        x_action = self.action_in_proj(x_action).unsqueeze(dim=1)

        # Concatenate obs and action and add positional encoding
        x = torch.cat([x_obs, x_action, self.reward_token.expand(n_samples, -1, -1)], dim=1)
        x = x + self.pos_embedding

        # Calculate η values
        attention_sum = torch.zeros(n_samples, self.input_token_len, self.input_token_len, device=x.device)
        eta = torch.eye(self.input_token_len, device=x.device).repeat(
            n_samples, 1, 1
        )

        # Apply transformer layers
        for layer in self.layers:
            x, attention_sum_layer, attention_mask_layer = layer(x)
            
            # Calculate attention sum for L1 regularization
            attention_sum += attention_sum_layer
            
            # Calculate η values and convert to binary
            eta += attention_mask_layer @ eta
            eta = eta.masked_fill_(eta != 0, 1.0)
            
        # Project obs and reward to output size (discard action token)
        pred_obs = self.obs_out_proj(x[:, :-2]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_proj(x[:, -1])

        return {
            "pred_obs": pred_obs,
            "pred_reward": pred_reward,
            "eta": eta,
            "attention_sum": attention_sum,
        }

class CombAction(nn.Module):
    def __init__(
        self,
        in_obs_shape: tuple[int, int, int],
        out_obs_shape: tuple[int, int, int],
        action_dim: int,
        d_model: int,  # Internal representation dimension
        n_heads: int,  # Number of attention heads
        n_layers: int,  # Number of transformer layers
        d_ff: int,  # Feed-forward network hidden dimension
        drop_prob: float = 0.1,  # Dropout probability
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]
        self.input_token_len = self.obs_token_len + 1  # obs + reward
        self.input_token_dim = in_obs_shape[-1] + action_dim

        # Input Projection
        self.in_proj = nn.Linear(self.input_token_dim, d_model)
        self.reward_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_token_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob)
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.obs_out_proj = nn.Linear(d_model, out_obs_shape[-1])
        self.reward_out_proj = nn.Linear(d_model, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack input
        x_obs = x[0].float()
        x_action = x[1].float()

        # Vars
        n_samples = x_obs.shape[0]

        # Create input sequence
        x = torch.zeros(n_samples, self.obs_token_len, self.input_token_dim, device=x_obs.device)
        x[..., : x_obs.shape[-1]] = x_obs.view(n_samples, self.obs_token_len, x_obs.shape[-1])
        x[..., x_obs.shape[-1] :] = x_action.unsqueeze(1).expand(n_samples, self.obs_token_len, x_action.shape[-1])

        # Embed input sequence
        x = self.in_proj(x)

        # Append reward token and add positional encoding
        x = torch.cat([x, self.reward_token.expand(n_samples, -1, -1)], dim=1)
        x = x + self.pos_embedding

        # Calculate η values
        attention_sum = torch.zeros(n_samples, self.input_token_len, self.input_token_len, device=x.device)
        eta = torch.eye(self.input_token_len, device=x.device).repeat(
            n_samples, 1, 1
        )

        # Apply transformer layers
        for layer in self.layers:
            x, attention_sum_layer, attention_mask_layer = layer(x)
            
            # Calculate attention sum for L1 regularization
            attention_sum += attention_sum_layer
            
            # Calculate η values and convert to binary
            eta += attention_mask_layer @ eta
            eta = eta.masked_fill_(eta != 0, 1.0)

        # Project obs and reward to output size
        pred_obs = self.obs_out_proj(x[:, :-1]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_proj(x[:, -1])

        return {
            "pred_obs": pred_obs,
            "pred_reward": pred_reward,
            "eta": eta,
            "attention_sum": attention_sum,
        }