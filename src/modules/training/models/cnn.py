import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, List


class CNN(nn.Module):
    """CNN model for predicting next observation and reward given current observation and action."""
    
    def __init__(
        self,
        hidden_dims: list[int] = [32, 64, 128],
        latent_dim: int = 32,
        use_batch_norm: bool = True,
        obs_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
    ):
        """Initialize the CNN model structure."""
        super().__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        
        self.obs_loss_weight = obs_loss_weight
        self.reward_loss_weight = reward_loss_weight
        
    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the model parameters from pipeline info."""
        self.info = info
        ti = info["token_index"]
        ti.discrete = True
        
        # Store softmax ranges for each grid cell component
        self.softmax_ranges = [
            ti.observation[0],
            ti.observation[1],
            ti.observation[2],
            ti.observation[3],
        ]
        
        # Set observation shape and action dimension from environment info
        self.obs_shape: Tuple[int, int, int] = (*info['env_build']['observation_space'].shape[:2], len(ti.observation_))
        self.action_dim: int = info['env_build']["action_space"].n.item()
        
        # For reward prediction, use number of classes if discrete
        self.reward_dim = ti.info['reward'][0][1] if ti.info['reward'][0][1] > 0 else 1
        
        # Initialize network architecture
        self._build_network()
        
        return info
    
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

    def _build_network(self) -> None:
        """Build the network architecture based on setup parameters."""
        # Calculate the spatial dimensions after encoding
        h, w, c = self.obs_shape
        self.encoded_height, self.encoded_width = self._calculate_conv_output_shape(h, w)

        
        # Encoder layers
        self.encode = nn.Sequential()
        in_channels = c
        for hidden_dim in self.hidden_dims:
            self.encode.append(
                nn.Conv2d(
                    in_channels, 
                    hidden_dim, 
                    kernel_size=3, 
                    stride=1,
                    padding=1)
                )
            self.encode.append(nn.ReLU())
            if self.use_batch_norm:
                self.encode.append(nn.BatchNorm2d(hidden_dim))
            in_channels = hidden_dim
                        
        # Action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, self.latent_dim),
            nn.ReLU()
        )
        
        # Fusion layer        
        flat_size = self.hidden_dims[-1] * self.encoded_height * self.encoded_width
        self.fusion = nn.Sequential(
            nn.Linear(flat_size + self.latent_dim, flat_size),
            nn.ReLU()
        )
        
        # Decoder layers
        reversed_dims = list(reversed(self.hidden_dims))
        self.decode = nn.Sequential()
        for i in range(len(reversed_dims) - 1):
            self.decode.append(
                nn.ConvTranspose2d(
                    reversed_dims[i],
                    reversed_dims[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0)
                )
            self.decode.append(nn.ReLU())
            if self.use_batch_norm:
                self.decode.append(nn.BatchNorm2d(reversed_dims[i + 1]))
            
        # Final observation decoder
        self.decode.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1],
                c,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0
            )
        ])
        
        # self.final_decoder = nn.ConvTranspose2d(
        #     reversed_dims[-1],
        #     c,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        # )
                
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(flat_size, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.reward_dim)
        )

    def _calculate_conv_output_shape(self, h: int, w: int) -> tuple[int, int]:
        """Calculate the output shape after all convolution layers.
        
        Args:
            h: Input height
            w: Input width
            
        Returns:
            tuple: (output_height, output_width)
        """
        def conv_output_shape(h_in: int, stride: int = 1, padding: int = 1, kernel_size: int = 3) -> int:
            return ((h_in + 2*padding - kernel_size) // stride) + 1
        
        # Apply the formula for each conv layer
        curr_h, curr_w = h, w
        for _ in self.hidden_dims:
            curr_h = conv_output_shape(curr_h)
            curr_w = conv_output_shape(curr_w)
            
        return curr_h, curr_w

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        
        # Unpack input
        obs = x[0].float()
        action = x[1].float()

        # Permute the observation to [B, C, H, W] format for Conv2d
        obs = obs.permute(0, 3, 1, 2)  # Now shape: [B, C, H, W]
        
        # Encode observation
        encoded_obs = self.encode(obs)
        batch_size = encoded_obs.size(0)
        
        # Flatten encoded observation
        # flat_encoded = encoded_obs.view(batch_size, -1)
        flat_encoded = encoded_obs.reshape(batch_size, -1)
        
        # Embed action
        embedded_action = self.action_embedding(action)
        
        # Fuse observation and action
        fused = self.fusion(torch.cat([flat_encoded, embedded_action], dim=1))
        
        # Reshape for decoder
        decoder_input = fused.view(
            batch_size,
            self.hidden_dims[-1],
            self.encoded_height,
            self.encoded_width
        )
        
        # Decode to next observation
        raw_next_obs = self.decode(decoder_input)
        
        # Permute back to original format
        raw_next_obs = raw_next_obs.permute(0, 2, 3, 1)  # Back to [B, H, W, C]
        
        # Apply softmax to appropriate ranges in the grid cells
        predicted_next_obs = self._apply_softmax_to_grid(raw_next_obs)
        
        # Predict reward
        raw_reward = self.reward_predictor(fused)
        predicted_reward = F.softmax(raw_reward, dim=-1) if self.reward_dim > 1 else raw_reward
        
        return predicted_next_obs, predicted_reward

    def compute_loss(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
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
                obs_loss += F.cross_entropy(
                    pred_range.reshape(-1, len(softmax_range)),
                    target_range.argmax(dim=-1).reshape(-1)
                )
            else:
                # For single values, use MSE
                pred_range = predicted_next_obs[..., softmax_range]
                target_range = target_next_obs[..., softmax_range]
                obs_loss += F.mse_loss(pred_range, target_range)
        
        # Compute reward loss based on discretization
        if self.reward_dim > 1:  # Discretized rewards
            reward_loss = F.cross_entropy(predicted_reward, target_reward.argmax(dim=-1))
        else:  # Continuous rewards
            reward_loss = F.mse_loss(predicted_reward, target_reward)
        
        # Compute weighted total loss
        total_loss = self.obs_loss_weight * obs_loss + self.reward_loss_weight * reward_loss
                
        return total_loss
    
    def get_dataset_cls(self):
        from ..datasets.two_d_dataset import TwoDDataset

        return functools.partial(TwoDDataset, discretize=True)
