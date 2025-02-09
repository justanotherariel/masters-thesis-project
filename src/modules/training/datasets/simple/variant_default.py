import torch

from src.typing.pipeline_objects import PipelineInfo

from .dataset import SimpleDataset, TensorIndex


class SDDefault(SimpleDataset):
    @staticmethod
    def create_ti(info: PipelineInfo, discrete: bool | None = None) -> TensorIndex:
        return SimpleDataset.create_base_ti(info, discrete)

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
        batch_size = len(batch)
        
        x_obs_dim = batch[0][0][0].shape
        x_obs_batch = torch.empty(batch_size, *x_obs_dim, dtype=batch[0][0][0].dtype)
        
        action_dim = batch[0][0][1].shape
        action_batch = torch.empty(batch_size, *action_dim, dtype=batch[0][0][1].dtype)

        y_obs_dim = batch[0][1][0].shape
        y_obs_batch = torch.empty(batch_size, *y_obs_dim, dtype=batch[0][1][0].dtype)
        
        reward_dim = batch[0][1][1].shape
        reward_batch = torch.empty(batch_size, *reward_dim, dtype=batch[0][1][1].dtype)
        
        for idx, ((x_obs, action), (y_obs, reward)) in enumerate(batch):
            x_obs_batch[idx] = x_obs        # (B, H, W, C)
            action_batch[idx] = action      # (B, A) or (B, num_classes)
            y_obs_batch[idx] = y_obs        # (B, H, W, C)
            reward_batch[idx] = reward      # (B, 1) or (B, num_classes)

        return (x_obs_batch, action_batch), (y_obs_batch, reward_batch)
