"""Transformer model with sparse attention mechanism."""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
    
class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, sparsity: float = 0.9):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.sparsity = sparsity
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

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

        output = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, sparsity=0.9):
        super().__init__()
        self.attn = SparseMultiHeadAttention(d_model, num_heads, dropout, sparsity)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
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
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float = 0.1
    sparsity: float = 0.9
    max_seq_length: int = 512

    def __post_init__(self):
        super().__init__()
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.d_model)
        )
        self._init_pos_encoding()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                SparseTransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout, self.sparsity)
                for _ in range(self.num_layers)
            ]
        )
        
    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        ti = info['token_index']
        ti.discrete = True
        self.softmax_ranges = [
            ti.type_,
            ti.observation[0],
            ti.observation[1],
            ti.observation[2],
            ti.action_,
        ]
        
        return info
        
    # def _init_pos_encoding(self):
    #     position = torch.arange(self.max_seq_length).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
    #     pe = torch.zeros(1, self.max_seq_length, self.d_model)
    #     pe[0, :, 0::2] = torch.sin(position * div_term)
    #     pe[0, :, 1::2] = torch.cos(position * div_term)
    #     self.pos_encoding.data.copy_(pe)
    
    def _init_pos_encoding(self):
        position = torch.arange(self.max_seq_length).unsqueeze(1)  # [seq_len, 1]
        
        # Calculate dimensions for even indices and odd indices
        even_dim = (self.d_model + 1) // 2  # Number of sin terms (ceiling division)
        odd_dim = self.d_model // 2         # Number of cos terms (floor division)
        
        # Create div_term for the maximum number of terms needed
        div_term = torch.exp(
            torch.arange(0, even_dim) * (-math.log(10000.0) / self.d_model)
        )
        
        # Initialize positional encoding tensor
        pe = torch.zeros(1, self.max_seq_length, self.d_model)
        
        # Generate indices for sin and cos terms
        even_indices = torch.arange(0, self.d_model, 2)  # [0, 2, 4, ...]
        odd_indices = torch.arange(1, self.d_model, 2)   # [1, 3, 5, ...]
        
        # Fill in sin terms (even indices)
        pe[0, :, even_indices] = torch.sin(position * div_term[:len(even_indices)])
        
        # Fill in cos terms (odd indices)
        if len(odd_indices) > 0:
            pe[0, :, odd_indices] = torch.cos(position * div_term[:len(odd_indices)])
        
        self.pos_encoding.data.copy_(pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate mask based on padding tokens in x sequence
        mask = (x[:, :, 0] == 0).unsqueeze(1).unsqueeze(1)

        # Modify input
        x = x.float() # Convert to float
        x = x + self.pos_encoding[:, :x.size(1)] # Positional encoding
        x = self.norm(x) # Normalize

        # Apply layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Apply softmax
        for range in self.softmax_ranges:
            x[:, :, range] = F.softmax(x[:, :, range], dim=-1)

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

    def get_dataset_cls(self):
        from ..datasets.token_dataset import TokenMinigridDataset

        return functools.partial(TokenMinigridDataset, discretize = True)
