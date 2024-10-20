"""Transformer model with sparse attention mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# Special tokens
SOS_TOKEN = torch.tensor([100, 0, 1], dtype=torch.int8)  # Start of sequence
SEP_TOKEN = torch.tensor([100, 0, 2], dtype=torch.int8)  # Separator
PAD_TOKEN = torch.tensor([100, 0, 0], dtype=torch.int8)  # Padding


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, sparsity=0.9):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.sparsity = sparsity

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = (
            self.W_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        key = (
            self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )
        value = (
            self.W_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply sparsity
        topk = int((1 - self.sparsity) * scores.size(-1))
        top_values, _ = torch.topk(scores, k=topk, dim=-1)
        threshold = top_values[..., -1, None]
        sparse_scores = scores * (scores >= threshold)

        attn = F.softmax(sparse_scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        output = (
            torch.matmul(attn, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.W_o(output)


class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, sparsity=0.9):
        super().__init__()
        self.attn = SparseMultiHeadAttention(d_model, num_heads, dropout, sparsity)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


@dataclass
class SparseTransformer(nn.Module):
    num_actions: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    input_dim: int = None
    dropout: float = 0.1
    sparsity: float = 0.9

    def __post_init__(self):
        super().__init__()

        if self.input_dim is not None:
            self.input_projection = nn.Linear(self.input_dim, self.d_model)
            self.output_projection = nn.Linear(self.d_model, self.input_dim)

        # self.norm = nn.BatchNorm1d(self.d_model if self.d_model is not None else self.input_dim)
        self.norm = nn.BatchNorm1d(52)

        self.layers = nn.ModuleList(
            [
                SparseTransformerBlock(
                    self.d_model, self.num_heads, self.d_ff, self.dropout, self.sparsity
                )
                for _ in range(self.num_layers)
            ]
        )

        self.action_embedding = nn.Embedding(self.num_actions, self.d_model)
        self.sos_token = nn.Parameter(torch.randn(self.d_model))
        self.sep_token = nn.Parameter(torch.randn(self.d_model))

    def forward(self, x):
        # Calculate mask based on padding tokens in x sequence
        mask = (x == PAD_TOKEN).all(dim=-1).unsqueeze(1).unsqueeze(1)

        # Get SOS and SEP tokens
        sos_mask = (x == SOS_TOKEN).all(dim=-1)
        sep_mask = (x == SEP_TOKEN).all(dim=-1)

        # Get action
        action = x[~sep_mask][:, -1].values

        # Project input to d_model dimensions
        if hasattr(self, "input_projection"):
            x = self.input_projection(x)

            # Replace SOS and SEP tokens
            x[sos_mask] = self.sos_token
            x[sep_mask] = self.sep_token

            # Embed action
            x[~sep_mask] = self.action_embedding(action)

        # Normalize x
        x = self.norm(x)

        # Apply layers
        for layer in self.layers:
            x = layer(x, mask)

        # Project back to input dimensions
        if hasattr(self, "output_projection"):
            x = self.output_projection(x)

        # Return last token
        return x[:, -1]

    def __repr__(self):
        original = super().__repr__()

        fields = self.__class__.__annotations__
        values = [(name, getattr(self, name)) for name in fields]
        class_vars = [f"{name}={value}" for name, value in values]

        return f"{original[:-2]}, {', '.join(class_vars)})"

    def __hash__(self):
        return hash(repr(self))
