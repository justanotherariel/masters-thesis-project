"""Transformer model adapted to RL transition models.

Heavily inspired by
    Author : Hyunwoong
    Github : https://github.com/hyunwoongko/transformer
"""

import functools
import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super().__init__(vocab_size, d_model, padding_idx=1)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

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

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
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


class ScaleDotProductAttention(nn.Module):
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


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        drop_prob=0.1,
        obs_loss_weight: float = 0.5,
        reward_loss_weight: float = 0.5,
    ):
        super().__init__()

        self.d_model = d_model  # Internal representation dimension
        self.n_heads = n_heads  # Number of attention heads
        self.n_layers = n_layers  # Number of transformer layers
        self.d_ff = d_ff  # Feed-forward network hidden dimension
        self.drop_prob = drop_prob  # Dropout probability

        self.obs_loss_weight = obs_loss_weight  # Weight for observation loss
        self.reward_loss_weight = reward_loss_weight  # Weight for reward loss

    def setup(self, info):
        self.info = info
        self.ti = info["token_index"]

        # Store softmax ranges for each grid cell component
        self.ti.discrete = True
        self.softmax_ranges = [
            self.ti.observation[0],
            self.ti.observation[1],
            self.ti.observation[2],
            self.ti.observation[3],
        ]

        # Set observation shape and action dimension from environment info
        self.input_dim: tuple[int, int] = (
            math.prod(info["env_build"]["observation_space"].shape[:2]) + 1,
            self.ti.shape,
        )
        self.output_dim: tuple[int, int] = (
            math.prod(info["env_build"]["observation_space"].shape[:2]) + 1,
            self.ti.shape,
        )

        # Set Network Parameters
        self.input_len = self.input_dim[0]
        self.input_size = self.input_dim[1]
        self.output_len = self.output_dim[0]
        self.output_size = self.output_dim[1]

        # Initialize network architecture
        self._build_network()

        info.update({"train": {"dataset": "TokenDataset"}})
        return info

    def _build_network(self) -> None:
        # Input embedding
        self.input_embedding = nn.Linear(self.input_size, self.d_model)
        self.input_pos_embedding = nn.Parameter(torch.randn(1, self.input_len, self.d_model))

        # Output embedding
        self.output_pos_embedding = nn.Parameter(torch.randn(1, self.output_len, self.d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=self.d_model, ffn_hidden=self.d_ff, n_head=self.n_heads, drop_prob=self.drop_prob
                )
                for _ in range(self.n_layers)
            ]
        )

        # Output projection
        self.output_linear = nn.Linear(self.d_model, self.output_size)

        # Save sizes
        self.input_len = self.input_len
        self.output_len = self.output_len

        self.dropout = nn.Dropout(p=self.drop_prob)

    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]

        # Embed input sequence
        x = self.input_embedding(x)  # [batch, input_len, d_model]
        x = x + self.input_pos_embedding

        # Create output positional tokens
        output_tokens = self.output_pos_embedding.expand(batch_size, -1, -1)  # [batch, output_len, d_model]

        # Concatenate input and output tokens
        x = torch.cat([x, output_tokens], dim=1)  # [batch, input_len + output_len, d_model]

        # Apply transformer layers
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)

        # Extract only the output sequence positions
        x = x[:, -self.output_len :, :]  # [batch, output_len, d_model]

        # Project to output size
        x = self.output_linear(x)  # [batch, output_len, output_size]

        return x

    def apply_softmax_to_tokens(self, x):
        """Apply softmax to each token in the output sequence.

        Args:
            x: Tensor of shape (batch_size, output_seq_len, token_dim)

        Returns:
            Tensor of same shape with softmax applied to appropriate ranges
        """

        # Create output tensor
        output = torch.zeros_like(x)

        # For each softmax range
        for softmax_range in self.softmax_ranges:
            if len(softmax_range) > 1:  # Only apply softmax if range has multiple elements
                # Extract the relevant slice
                sliced = x[..., softmax_range]

                # Apply softmax along the last dimension
                softmaxed = F.softmax(sliced, dim=-1)

                # Place back in output
                output[..., softmax_range] = softmaxed
            else:
                # For single values (no softmax needed), just copy
                output[..., softmax_range] = x[..., softmax_range]

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        x = x.float()
        y = y.float()

        predicted_next_obs = x[:, :-1]
        predicted_reward = x[:, -1, self.ti.reward_].squeeze()

        target_next_obs = y[:, :-1]
        target_reward = y[:, -1, self.ti.reward_].squeeze()

        obs_loss = 0

        for softmax_range in self.softmax_ranges:
            if len(softmax_range) > 1:
                # For softmax ranges, use cross entropy loss
                pred_range = predicted_next_obs[..., softmax_range]
                target_range = target_next_obs[..., softmax_range]
                loss = F.cross_entropy(
                    pred_range.reshape(-1, len(softmax_range)), target_range.argmax(dim=-1).reshape(-1)
                )
            else:
                # For single values, use MSE
                pred_range = predicted_next_obs[..., softmax_range]
                target_range = target_next_obs[..., softmax_range]
                loss = F.mse_loss(pred_range, target_range)
            obs_loss += loss / len(softmax_range)

        # Compute reward loss
        reward_loss = F.mse_loss(predicted_reward, target_reward)

        # Combine losses
        total_loss = self.obs_loss_weight * obs_loss + self.reward_loss_weight * reward_loss

        return total_loss

    def get_dataset_cls(self):
        from ..datasets.token_dataset import TokenDataset

        return functools.partial(TokenDataset, discretize=True)
