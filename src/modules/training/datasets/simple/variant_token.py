import torch

from .dataset import SimpleDataset, TensorIndex
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

class SDToken(SimpleDataset):
    
    def setup(self, info: PipelineInfo) -> PipelineInfo:
        super().setup(info)
        
        self.obs_len = self._data.observations.shape[1] * self._data.observations.shape[2]
        self.action_len = 1
        self.reward_len = 1
        
        self.input_seq_shape = (self.obs_len + self.action_len, self.ti.observation_.shape[0] + self.ti.action_.shape[0])
        self.output_seq_shape = (self.obs_len + self.reward_len, self.ti.observation_.shape[0] + self.ti.reward_.shape[0])

    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        (x_obs, x_action), (y_obs, y_reward) = super().__getitem__(idx)
        
        input_seq = torch.zeros(self.input_seq_shape, dtype=torch.uint8)
        input_seq[:self.obs_len, self.ti.observation_] = x_obs.view(-1, self.ti.observation_.shape[0])
        input_seq[self.obs_len:, self.ti.action_] = x_action.view(-1, self.ti.action_.shape[0])
        
        output_seq = torch.zeros(self.output_seq_shape, dtype=torch.uint8)
        output_seq[:self.obs_len, self.ti.observation_] = y_obs.view(-1, self.ti.observation_.shape[0])
        output_seq[self.obs_len:, self.ti.reward_] = y_reward.view(-1, self.ti.reward_.shape[0]).to(torch.uint8)
        
        return input_seq, output_seq
        
    
    @staticmethod
    def create_ti(info: PipelineInfo) -> TensorIndex:
        """Create a TokenIndex object from the given info dictionary."""
        observation_info = info.data_info["observation_info"]
        action_info = info.data_info["action_info"]
        reward_info = info.data_info["reward_info"]

        token_info = {}
        start_idx = 0

        token_info.update(
            {
                "observation": [(start_idx + idx, num_items) for (idx, num_items) in observation_info],
            }
        )
        start_idx += len(observation_info)

        token_info.update(
            {
                "action": [(start_idx + idx, num_items) for (idx, num_items) in action_info],
            }
        )
        token_info.update(
            {
                "reward": [(start_idx + idx, num_items) for (idx, num_items) in reward_info],
            }
        )

        return TensorIndex(token_info)
