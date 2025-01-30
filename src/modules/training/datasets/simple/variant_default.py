import torch

from .dataset import SimpleDataset, TensorIndex
from src.typing.pipeline_objects import PipelineInfo

class SDDefault(SimpleDataset):    
    @staticmethod
    def create_ti(info: PipelineInfo) -> TensorIndex:
        return SimpleDataset.create_base_ti(info)

    @staticmethod
    def collate_fn(
        batch: list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Custom collate function for TwoDDataset that properly batches 2D observations with actions/rewards.

        Args:
            batch: List of ((x_obs, action), (y_obs, reward)) tuples
                - x_obs: shape (height, width, channels)
                - action: shape (action_dim,) or (num_classes,) if discretized
                - y_obs: shape (height, width, channels)
                - reward: shape (1,) or (num_classes,) if discretized

        Returns:
            tuple: ((x_obs_batch, action_batch), (y_obs_batch, reward_batch))
                - x_obs_batch: shape (batch_size, height, width, channels)
                - action_batch: shape (batch_size, action_dim) or (batch_size, num_classes)
                - y_obs_batch: shape (batch_size, height, width, channels)
                - reward_batch: shape (batch_size, 1) or (batch_size, num_classes)
        """
        # Unzip the batch into separate lists
        x_obs_list = []
        action_list = []
        y_obs_list = []
        reward_list = []

        for (x_obs, action), (y_obs, reward) in batch:
            x_obs_list.append(x_obs)
            action_list.append(action)
            y_obs_list.append(y_obs)
            reward_list.append(reward)

        # Stack the observations
        x_obs_batch = torch.stack(x_obs_list)  # (B, H, W, C)
        y_obs_batch = torch.stack(y_obs_list)  # (B, H, W, C)

        # Stack actions and rewards
        action_batch = torch.stack(action_list)  # (B, A) or (B, num_classes)
        reward_batch = torch.stack(reward_list)  # (B, 1) or (B, num_classes)

        return (x_obs_batch, action_batch), (y_obs_batch, reward_batch)
