"""U-Net model adapted to RL transition models.

Heavily inspired by:
  Author: milesial
  Github: https://github.com/milesial/Pytorch-UNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        in_obs_shape: tuple[int, int, int],
        out_obs_shape: tuple[int, int, int],
        action_dim: int,
        hidden_channels: list[int],
    ):
        """Initialize the CNN model structure."""
        super().__init__()

        if in_obs_shape[:2] != out_obs_shape[:2]:
            raise ValueError("Input and output observation shapes must have the same height and width.")

        self.hidden_channels = hidden_channels

        # Spatial Action
        n_fields = in_obs_shape[0] * in_obs_shape[1]
        self.spatial_action = nn.Sequential(
            nn.Linear(action_dim, n_fields),
            nn.ReLU(),
            nn.Linear(n_fields, n_fields)
        )

        # Down | First +1 channel for spatial action
        self.down_first = Down(in_obs_shape[-1] + 1, self.hidden_channels[0], first=True)
        self.down_layers = nn.ModuleList(
            [Down(self.hidden_channels[i], self.hidden_channels[i + 1]) for i in range(len(self.hidden_channels) - 1)]
        )

        # Up
        self.up_layers = nn.ModuleList(
            [
                Up(self.hidden_channels[i], self.hidden_channels[i - 1])
                for i in range(len(self.hidden_channels) - 1, 0, -1)
            ]
        )
        self.up_last = OutConv(self.hidden_channels[0], out_obs_shape[-1])
        
        # Reward
        self.reward = nn.Sequential(
            nn.Linear(out_obs_shape[0] * out_obs_shape[1] * out_obs_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""

        # Unpack input
        x_obs = x[0].float()
        x_action = x[1].float()
        
        # Concatenate spatial action to observation
        x_action = self.spatial_action(x_action).reshape(x_action.shape[0], x_obs.shape[1], x_obs.shape[2]).unsqueeze(-1)
        x_obs = torch.cat([x_obs, x_action], dim=-1)

        # Permute to (batch_size, channels, height, width)
        pred_obs = x_obs.permute(0, 3, 1, 2)

        # Down
        pred_obs = self.down_first(pred_obs)
        down_out = [pred_obs]
        for layer in self.down_layers:
            pred_obs = layer(pred_obs)
            down_out.append(pred_obs)
        down_out = down_out[:-1]  # Discard last output

        # Up
        for layer, down in zip(self.up_layers, down_out[::-1]):
            pred_obs = layer(pred_obs, down)
        pred_obs = self.up_last(pred_obs)

        # Permute obs back to (batch_size, height, width, channels)
        pred_obs = pred_obs.permute(0, 2, 3, 1)
        
        # Calculate Reward
        pred_reward = self.reward(pred_obs.reshape(pred_obs.size(0), -1))

        return pred_obs, pred_reward
