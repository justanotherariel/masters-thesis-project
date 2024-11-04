import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Special tokens - using an enum would be cleaner
from enum import IntEnum

class SpecialTokens(IntEnum):
    PAD = 0
    SOS = 1
    SEP = 2
    ACTION = 3
    REWARD = 4

SPECIAL_TOKENS = {
    SpecialTokens.PAD: torch.tensor([100, 0, 0], dtype=torch.uint8),
    SpecialTokens.SOS: torch.tensor([100, 0, 1], dtype=torch.uint8),
    SpecialTokens.SEP: torch.tensor([100, 0, 2], dtype=torch.uint8),
    SpecialTokens.ACTION: torch.tensor([100, 1, 0], dtype=torch.uint8),
    SpecialTokens.REWARD: torch.tensor([100, 2, 0], dtype=torch.uint8)
}

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, sparsity: float = 0.9):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.sparsity = sparsity
        self.d_k = d_model // num_heads
        
        # Combined projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Efficient combined projection
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        query, key, value = [
            layer.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer in qkv
        ]

        # Scaled dot-product attention with sparsity
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Improved sparse attention with stable softmax
        if self.sparsity > 0:
            topk = max(1, int((1 - self.sparsity) * scores.size(-1)))
            top_values, _ = torch.topk(scores, k=topk, dim=-1)
            threshold = top_values[..., -1, None]
            sparse_scores = scores * (scores >= threshold)
            
            # Stability improvements for sparse attention
            max_score = torch.max(sparse_scores, dim=-1, keepdim=True)[0]
            exp_scores = torch.exp(sparse_scores - max_score)
            exp_scores = exp_scores * (sparse_scores != -float('inf'))
            attn = exp_scores / (torch.sum(exp_scores, dim=-1, keepdim=True) + 1e-9)
        else:
            attn = F.softmax(scores, dim=-1)
        
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Compute output
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, sparsity: float = 0.9):
        super().__init__()
        self.attn = SparseMultiHeadAttention(d_model, num_heads, dropout, sparsity)
        
        # Improved feed-forward network with GELU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Pre-norm architecture for better training stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        normed_x = self.norm1(x)
        x = x + self.dropout(self.attn(normed_x, mask))
        
        normed_x = self.norm2(x)
        x = x + self.dropout(self.ff(normed_x))
        return x


@dataclass
class SparseTransformer(nn.Module):
    num_actions: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    input_dim: Optional[int] = None
    dropout: float = 0.1
    sparsity: float = 0.9
    max_seq_length: int = 512

    def __post_init__(self):
        super().__init__()
        
        # Input/output projections if dimensions differ
        if self.input_dim is not None:
            self.input_projection = nn.Linear(self.input_dim, self.d_model)
            self.output_projection = nn.Linear(self.d_model, self.input_dim)
            
            # Proper initialization
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.xavier_uniform_(self.output_projection.weight)

        # Layer normalization instead of batch norm for better stability
        self.norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.d_model)
        )
        self._init_pos_encoding()

        # Transformer layers
        self.layers = nn.ModuleList([
            SparseTransformerBlock(
                self.d_model, 
                self.num_heads, 
                self.d_ff, 
                self.dropout, 
                self.sparsity
            ) for _ in range(self.num_layers)
        ])

        # Token embeddings
        self.action_embedding = nn.Embedding(self.num_actions, self.d_model)
        self.sos_token = nn.Parameter(torch.randn(self.d_model))
        self.sep_token = nn.Parameter(torch.randn(self.d_model))
        
        # Initialize special tokens
        nn.init.normal_(self.sos_token, mean=0.0, std=0.02)
        nn.init.normal_(self.sep_token, mean=0.0, std=0.02)

    def _init_pos_encoding(self):
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, self.max_seq_length, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create attention mask for padding
        mask = (x != SPECIAL_TOKENS[SpecialTokens.PAD]).all(dim=-1).unsqueeze(1).unsqueeze(2)
        
        if hasattr(self, "input_projection"):
            # Handle special tokens
            sos_mask = (x == SPECIAL_TOKENS[SpecialTokens.SOS]).all(dim=-1)
            sep_mask = (x == SPECIAL_TOKENS[SpecialTokens.SEP]).all(dim=-1)
            action_indices = torch.nonzero(sep_mask)[:, 1] - 1
            actions = x[torch.arange(x.shape[0], device=x.device), action_indices][:, 2].to(torch.int)

            # Convert input to float and project
            x = self.input_projection(x.float())

            # Replace special tokens with learned embeddings
            x[sos_mask] = self.sos_token
            x[sep_mask] = self.sep_token
            x[torch.arange(x.shape[0], device=x.device), action_indices] = self.action_embedding(actions)
        else:
            x = x.float()

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Apply transformer layers with residual connections
        for layer in self.layers:
            x = layer(x, mask)
            
        # Project back to input dimensions if needed
        if hasattr(self, "output_projection"):
            x = self.output_projection(x)
            
        return x[:, -1]