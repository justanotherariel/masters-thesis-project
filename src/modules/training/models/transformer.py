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
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None = None
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
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))

        # 3. Convert scores to probabilities
        attention_probs = self.softmax(attention_scores)

        # 4. Apply attention dropout
        attention_probs_drop = self.dropout(attention_probs)

        # 5. Compute weighted values
        weighted_values = attention_probs_drop @ value

        return weighted_values, attention_probs


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_p: float = 0.1,
        use_bias: bool = True,
        eigenvalue_update_freq: int = 100,
        weight_change_threshold: float = 0.05,
    ):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.eigenvalue_update_freq = eigenvalue_update_freq
        self.weight_change_threshold = weight_change_threshold

        self.attention = ScaledDotProductAttention(attention_dropout=drop_p)

        # Linear projections
        self.query_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=use_bias)

        # Cache for eigenvalues and tracking variables
        self.register_buffer('max_eigenvalues', torch.ones(n_heads), persistent=False)
        self.register_buffer('last_value_weights', None, persistent=False)
        self.eigenvalues_computed = False
        self.forward_count = 0
        
    def compute_max_eigenvalues(self):
        """Compute the maximum eigenvalue for each head's value projection."""
        with torch.no_grad():
            weight = self.value_projection.weight  # shape: [d_model, d_model]
            d_k = self.d_model // self.n_heads
            
            # Save current weights for future comparison
            if self.last_value_weights is None or self.last_value_weights.shape != weight.shape:
                self.last_value_weights = weight.clone()
            else:
                self.last_value_weights.copy_(weight)
            
            # Initialize eigenvalues tensor
            max_eigenvalues = torch.zeros(self.n_heads, device=weight.device)
            
            for h in range(self.n_heads):
                # Extract the weight matrix for this head
                head_weight = weight[h*d_k:(h+1)*d_k, :]  # shape: [d_k, d_model]
                
                # Compute singular values (square roots of eigenvalues of W*W^T)
                _, s, _ = torch.linalg.svd(head_weight, full_matrices=False)
                max_eigenvalues[h] = torch.max(s)
            
            self.max_eigenvalues.copy_(max_eigenvalues)
            self.eigenvalues_computed = True
            print(f"Updated eigenvalues: {self.max_eigenvalues}")

    def check_weight_change(self) -> float:
        """Measure how much value weights have changed since last eigenvalue computation."""
        if self.last_value_weights is None:
            return float('inf')  # Force update if no previous weights stored
        
        current_weights = self.value_projection.weight
        # Relative Frobenius norm of the difference
        diff_norm = torch.norm(current_weights - self.last_value_weights)
        weights_norm = torch.norm(self.last_value_weights)
        relative_change = diff_norm / (weights_norm + 1e-8)
        
        return relative_change.item()

    def forward(
        self, x: torch.Tensor, prev_eta: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Applies multi-headed attention to the input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
            prev_eta: Optional previous eta values
            attention_mask: Optional attention mask
            force_eigenvalue_update: Force recomputation of eigenvalues

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
        attention_output, attention_weights = self.attention(
            query_heads, key_heads, value_heads, attention_mask=attention_mask
        )

        # 4. Merge attention heads and project to output dimension
        merged_attention = self.merge_heads(attention_output)
        output = self.output_projection(merged_attention)

        # 5. Calculate new η values if requested
        new_eta = None
        if prev_eta is not None:
            # Check if we should update eigenvalues based on various criteria
            should_update = False
                    
            # Criterion 1: First computation
            if not self.eigenvalues_computed:
                should_update = True
                
            # Only check the following criteria during training
            elif self.training:
                # Criterion 2: Update every N forward passes
                self.forward_count += 1
                if self.forward_count >= self.eigenvalue_update_freq:
                    should_update = True
                    self.forward_count = 0
                
                # Criterion 3: Significant weight change detected
                elif self.forward_count % 10 == 0:  # Check less frequently for efficiency
                    weight_change = self.check_weight_change()
                    if weight_change > self.weight_change_threshold:
                        should_update = True
                        self.forward_count = 0
            
            if should_update:
                self.compute_max_eigenvalues()

            # Scale attention weights by the maximum eigenvalue for each head
            scaled_weights = attention_weights * self.max_eigenvalues.view(1, self.n_heads, 1, 1)
            
            # Sum across heads to get the total influence
            eta_layer = scaled_weights.sum(dim=1)
            
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
        self.attention = MultiHeadedAttention(d_model=d_model, n_heads=n_heads, drop_p=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(
        self, x, prev_eta: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Performs a single transformer layer operation.

        Args:
            x (torch.Tensor): Input Tensor
            prev_eta (torch.Tensor | None, optional): Previous Eta. Defaults to None.
            attention_mask (torch.Tensor | None, optional): Attention Mask. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: output tensor and new eta
        """

        # Self attention
        delta_x, eta = self.attention(x, prev_eta=prev_eta, attention_mask=attention_mask)
        delta_x = self.dropout1(delta_x)
        x = self.norm1(x + delta_x)

        # Feed forward
        delta_x = self.ffn(x)
        delta_x = self.dropout2(delta_x)
        x = self.norm2(x + delta_x)

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
        calculate_eta: bool = False,  # Whether to calculate η values
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.calculate_eta = calculate_eta

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

        # Embed input sequence
        x_obs = x_obs.view(n_samples, self.obs_token_len, self.in_obs_shape[-1])
        x_obs = self.obs_in_proj(x_obs)
        x_action = self.action_in_proj(x_action).unsqueeze(dim=1)

        # Concatenate obs and action and add positional encoding
        x = torch.cat([x_obs, x_action, self.reward_token.expand(n_samples, 1, -1)], dim=1)
        x = x + self.pos_embedding

        # Calculate η values if requested
        eta = None
        if self.calculate_eta:
            eta = torch.eye(self.input_token_len, device=x.device).expand(
                n_samples, self.input_token_len, self.input_token_len
            )

        # Apply transformer layers
        for layer in self.layers:
            x, eta = layer(x, prev_eta=eta)

        # Project obs and reward to output size (discard action token)
        pred_obs = self.obs_out_proj(x[:, :-2]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_proj(x[:, -1])

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
        calculate_eta: bool = False,  # Whether to calculate η values
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.calculate_eta = calculate_eta

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
        x = torch.cat([x, self.reward_token.expand(n_samples, 1, -1)], dim=1)
        x = x + self.pos_embedding

        # Calculate η values if requested
        eta = None
        if self.calculate_eta:
            eta = torch.eye(self.input_token_len, device=x.device).expand(
                n_samples, self.input_token_len, self.input_token_len
            )

        # Apply transformer layers
        for layer in self.layers:
            x, eta = layer(x, prev_eta=eta)

        # Project obs and reward to output size
        pred_obs = self.obs_out_proj(x[:, :-1]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_proj(x[:, -1])

        return (pred_obs, pred_reward) if eta is None else (pred_obs, pred_reward, eta)
