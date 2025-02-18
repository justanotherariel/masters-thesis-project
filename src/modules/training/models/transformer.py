"""Transformer model adapted to RL transition models.

Heavily inspired by
    Author : Hyunwoong
    Github : https://github.com/hyunwoongko/transformer
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from typing import Optional, Tuple
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_heads
        tensor = tensor.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
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
        self.attention = MultiHeadedAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Self attention
        _x = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class TransformerSepAction(nn.Module):
    def __init__(
        self,
        in_obs_shape: tuple[int, int, int],
        out_obs_shape: tuple[int, int, int],
        action_dim: int,
        d_model: int,    # Internal representation dimension
        n_heads: int,   # Number of attention heads
        n_layers: int,  # Number of transformer layers
        d_ff: int,  # Feed-forward network hidden dimension
        drop_prob: float = 0.1, # Dropout probability
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]

        # Input Projection and Positional Encoding
        self.obs_in_up = nn.Linear(in_obs_shape[-1], d_model)
        self.action_in_up = nn.Linear(action_dim, d_model)
        self.input_pos_embedding = nn.Parameter(torch.randn(1, self.obs_token_len +1, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob
                )
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

        # Embed input sequence
        x_obs = x_obs.view(x_obs.shape[0], self.obs_token_len, self.in_obs_shape[-1])
        x_obs = self.obs_in_up(x_obs)
        x_action = self.action_in_up(x_action).unsqueeze(dim=1)
        
        # Concatenate obs and action and add positional encoding
        x = torch.cat([x_obs, x_action], dim=1)
        x = x + self.input_pos_embedding

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Project to output size
        pred_obs = self.obs_out_down(x[:, :-1]).view(-1, *self.out_obs_shape)
        pred_reward = self.reward_out_down(x[:, -1])

        return pred_obs, pred_reward

class TransformerCombAction(nn.Module):
    def __init__(
        self,
        in_obs_shape: tuple[int, int, int],
        out_obs_shape: tuple[int, int, int],
        action_dim: int,
        d_model: int,    # Internal representation dimension
        n_heads: int,   # Number of attention heads
        n_layers: int,  # Number of transformer layers
        d_ff: int,  # Feed-forward network hidden dimension
        drop_prob: float = 0.1, # Dropout probability
    ):
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.in_obs_shape = in_obs_shape
        self.out_obs_shape = out_obs_shape
        self.obs_token_len = in_obs_shape[0] * in_obs_shape[1]
        
        self.input_token_dim = in_obs_shape[-1] + action_dim

        # Input Projection and Positional Encoding
        self.in_up = nn.Linear(self.input_token_dim, d_model)
        self.input_pos_embedding = nn.Parameter(torch.randn(1, self.obs_token_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model, ffn_hidden=d_ff, n_heads=n_heads, drop_prob=drop_prob
                )
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
        x[..., :x_obs.shape[-1]] = x_obs.view(n_samples, self.obs_token_len, x_obs.shape[-1])
        x[..., x_obs.shape[-1]:] = x_action.unsqueeze(1).expand(n_samples, self.obs_token_len, x_action.shape[-1])
        
        # Embed token sequence and add positional encoding
        x = self.in_up(x)
        x = x + self.input_pos_embedding

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Extract obs and reward
        pred_obs = x[..., :x_obs.shape[-1]].reshape(n_samples, *self.out_obs_shape)
        pred_reward = x[..., -1].mean(dim=-1, keepdim=True)

        return pred_obs, pred_reward
