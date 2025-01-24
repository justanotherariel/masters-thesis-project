"""U-Net model adapted to RL transition models.

Heavily inspired by:
  Author: milesial
  Github: https://github.com/milesial/Pytorch-UNet/
"""

import functools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.typing.pipeline_objects import PipelineInfo


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()

        self.maxpool_conv = nn.Sequential(DoubleConv(in_channels, out_channels))

        if not first:
            self.maxpool_conv.insert(0, nn.MaxPool2d(2))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.size(), x2.size())

        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """A modified U-Net model for predicting next observation and reward given current observation and action."""

    def __init__(
        self,
        hidden_channels: list[int],
        obs_loss_weight: float = 0.5,
        reward_loss_weight: float = 0.5,
    ):
        """Initialize the CNN model structure."""
        super().__init__()

        self.hidden_channels = hidden_channels
        self.obs_loss_weight = obs_loss_weight
        self.reward_loss_weight = reward_loss_weight

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        """Setup the model parameters from pipeline info."""
        self.info = info
        ti = info.model_ti

        # Store softmax ranges for each grid cell component
        ti.discrete = True
        self.softmax_ranges = [
            ti.observation[0],
            ti.observation[1],
            ti.observation[2],
            ti.observation[3],
        ]

        # Set observation shape and action dimension from environment info
        self.obs_dim: tuple[int, int, int] = (*info.data_info["observation_space"].shape[:2], len(ti.observation_))
        self.action_dim: int = info.data_info["action_space"].n.item()

        # Initialize network architecture
        self._build_network()

        # info.model_ds_class = "TwoDDataset"
        return info

    def _build_network(self) -> None:
        """Build the network architecture based on setup parameters."""

        # Down
        self.down_first = Down(self.obs_dim[-1], self.hidden_channels[0], first=True)
        self.down_layers = nn.ModuleList(
            [Down(self.hidden_channels[i], self.hidden_channels[i + 1]) for i in range(len(self.hidden_channels) - 1)]
        )

        # Fusion
        final_encoder_size_y = self.obs_dim[0] // 2 ** (len(self.hidden_channels) - 1)
        final_encoder_size_x = self.obs_dim[1] // 2 ** (len(self.hidden_channels) - 1)
        final_encode_size_flat = final_encoder_size_y * final_encoder_size_x * self.hidden_channels[-1]
        self.fusion = nn.Linear(final_encode_size_flat + self.action_dim, final_encode_size_flat)

        # Reward
        self.reward = nn.Linear(final_encode_size_flat, 1)

        # Up
        self.up_layers = nn.ModuleList(
            [
                Up(self.hidden_channels[i], self.hidden_channels[i - 1])
                for i in range(len(self.hidden_channels) - 1, 0, -1)
            ]
        )
        self.up_last = OutConv(self.hidden_channels[0], self.obs_dim[-1])

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""

        # Unpack input
        obs = x[0].float()
        action = x[1].float()

        # Permute to (batch_size, channels, height, width)
        obs = obs.permute(0, 3, 1, 2)

        # Down
        obs = self.down_first(obs)
        down_out = [obs]
        for layer in self.down_layers:
            obs = layer(obs)
            down_out.append(obs)
        down_out = down_out[:-1]  # Discard last output

        # Flatten obs
        obs_flat = obs.reshape(obs.size(0), -1)

        # Fuse Action with final up output
        action = action.view(action.size(0), -1)
        obs_flat = self.fusion(torch.cat([obs_flat, action], dim=1))

        # Calculate Reward
        reward = self.reward(obs_flat)

        # Reshape obs back to (batch_size, channels, height, width)
        obs = obs.view(obs.size(0), self.hidden_channels[-1], obs.size(2), obs.size(3))

        # Up
        for layer, down in zip(self.up_layers, down_out[::-1]):
            obs = layer(obs, down)
        obs = self.up_last(obs)

        # Permute obs back to (batch_size, height, width, channels)
        obs = obs.permute(0, 2, 3, 1)

        # Apply softmax for discrete values
        predicted_next_obs = self._apply_softmax_to_grid(obs)

        return predicted_next_obs, reward

    def _apply_softmax_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax to each range in the grid cells.

        Args:
            x: Tensor of shape (batch_size, height, width, channels)

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
        x: tuple[torch.Tensor, torch.Tensor],
        y: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """Compute the combined loss for observation and reward predictions."""
        # Compute observation loss using cross entropy for softmaxed ranges
        predicted_next_obs, predicted_reward = x
        target_next_obs, target_reward = y

        obs_loss = 0

        for softmax_range in self.softmax_ranges:
            if len(softmax_range) > 1:  # If it's a softmax range
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

        # Compute weighted total loss
        total_loss = self.obs_loss_weight * obs_loss + self.reward_loss_weight * reward_loss

        return total_loss

    def get_dataset_cls(self):
        from ..datasets.two_d_dataset import TwoDDataset

        return functools.partial(TwoDDataset, discretize=True)
