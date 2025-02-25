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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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

        if not (query.dim() == key.dim() == value.dim() == 4):
            raise RuntimeError("Query, Key, and Value must be 4-dimensional tensors")
            
        batch_size, n_heads, seq_length, dim_per_head = key.size()
        
        if query.size(3) != dim_per_head:
            raise RuntimeError(f"Query dimension {query.size(3)} doesn't match Key dimension {dim_per_head}")

        # 1. Compute attention scores
        key_transpose = key.transpose(2, 3)  # transpose
        attention_scores = (query @ key_transpose) / math.sqrt(dim_per_head)

        # 2. Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))

        # 3. Convert scores to probabilities
        attention_probs = self.softmax(attention_scores)
        
        # 4. Apply attention dropout
        attention_probs = self.dropout(attention_probs)
        
        # 5. Compute weighted values
        weighted_values = attention_probs @ value

        return weighted_values, attention_probs


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention_dropout: float = 0.1, use_bias: bool = True):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.attention = ScaledDotProductAttention(attention_dropout=attention_dropout)
        
        # Linear projections
        self.query_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=use_bias)
        # self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)
            nn.init.zeros_(self.key_projection.bias)
            nn.init.zeros_(self.value_projection.bias)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor, prev_eta: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Applies multi-headed attention to the input tensor.
        
        Args:
            input_tensor: Tensor of shape [batch_size, seq_length, d_model]
            attention_mask: Optional attention mask
            is_training: Whether the model is in training mode
            
        Returns:
            tuple: (output_tensor, new_eta)
        """
        batch_size, seq_length, _ = x.size()

        # 1. Project input into query, key, and value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # 2. Split projections into multiple heads
        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(value)

        # 3. Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(
            query_heads, key_heads, value_heads, 
            attention_mask=attention_mask
        )

        # 4. Merge attention heads and project to output dimension
        merged_attention = self.merge_heads(attention_output)
        output = self.output_projection(merged_attention)
        
        # 5. Calculate new η values if requested
        new_eta = None
        if prev_eta is not None:
            # Average attention weights across heads
            eta_layer = attention_weights.mean(dim=1)
                        
            # Update η using matrix multiplication
            new_eta = torch.bmm(eta_layer, prev_eta)

        return output, new_eta

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
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=d_model, n_heads=n_heads, attention_dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, prev_eta: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Performs a single transformer layer operation.

        Args:
            x (torch.Tensor): Input Tensor
            prev_eta (torch.Tensor | None, optional): Previous Eta. Defaults to None.
            attention_mask (torch.Tensor | None, optional): Attention Mask. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: output tensor and new eta
        """
        
        # Self attention
        _x = x
        x, eta = self.attention(x, prev_eta=prev_eta, attention_mask=attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, eta


class TransformerSepAction(nn.Module):
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
        calculate_eta: bool = False,    # Whether to calculate η values
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]
        self.input_token_len = self.obs_token_len + 1
        
        self.calculate_eta = calculate_eta

        # Input Projection and Positional Encoding
        self.obs_in_up = nn.Linear(in_obs_shape[-1], d_model)
        self.action_in_up = nn.Linear(action_dim, d_model)
        self.input_pos_embedding = nn.Parameter(torch.randn(1, self.input_token_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob)
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.obs_out_down = nn.Linear(d_model, out_obs_shape[-1])
        self.reward_out_down = nn.Linear(d_model, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack input
        x_obs = x[0].float()
        x_action = x[1].float()
        
        # Vars
        n_samples = x_obs.shape[0]

        # Embed input sequence
        x_obs = x_obs.view(n_samples, self.obs_token_len, self.in_obs_shape[-1])
        x_obs = self.obs_in_up(x_obs)
        x_action = self.action_in_up(x_action).unsqueeze(dim=1)

        # Concatenate obs and action and add positional encoding
        x = torch.cat([x_obs, x_action], dim=1)
        x = x + self.input_pos_embedding

        # Apply transformer layers
        eta = torch.eye(self.input_token_len, device=x.device).expand(n_samples, self.input_token_len, self.input_token_len) if self.calculate_eta else None
        for layer in self.layers:
            x, eta = layer(x, prev_eta=eta)

        # Project to output size
        pred_obs = self.obs_out_down(x[:, :-1]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_down(x[:, -1])

        return (pred_obs, pred_reward) if eta is None else (pred_obs, pred_reward, eta)


class TransformerCombAction(nn.Module):
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
        calculate_eta: bool = False,    # Whether to calculate η values
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]
        self.input_token_len = self.obs_token_len
        self.input_token_dim = in_obs_shape[-1] + action_dim

        self.calculate_eta = calculate_eta

        # Input Projection and Positional Encoding
        self.in_up = nn.Linear(self.input_token_dim, d_model)
        self.input_pos_embedding = nn.Parameter(torch.randn(1, self.input_token_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob)
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.out_down = nn.Linear(d_model, out_obs_shape[-1])

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack input
        x_obs = x[0].float()
        x_action = x[1].float()

        # Vars
        n_samples = x_obs.shape[0]

        # Reshape the input to a token sequence
        # Expand action and add to each obs token
        x = torch.zeros(n_samples, self.obs_token_len, self.input_token_dim, device=x_obs.device)
        x[..., : x_obs.shape[-1]] = x_obs.view(n_samples, self.obs_token_len, x_obs.shape[-1])
        x[..., x_obs.shape[-1] :] = x_action.unsqueeze(1).expand(n_samples, self.obs_token_len, x_action.shape[-1])

        # Embed token sequence and add positional encoding
        x = self.in_up(x)
        x = x + self.input_pos_embedding

        # Apply transformer layers
        eta = torch.eye(self.input_token_len, device=x.device).expand(n_samples, self.input_token_len, self.input_token_len) if self.calculate_eta else None
        for layer in self.layers:
            x, eta = layer(x, prev_eta=eta)

        # Extract obs and reward
        pred_obs = x[..., : x_obs.shape[-1]].reshape(n_samples, *self.out_obs_shape)
        pred_reward = x[..., -1].mean(dim=-1, keepdim=True)

        return (pred_obs, pred_reward) if eta is None else (pred_obs, pred_reward, eta)
