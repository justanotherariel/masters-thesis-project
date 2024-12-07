from typing import Any

from src.framework.transforming import TransformationBlock
from src.modules.training.datasets.utils import TokenIndex
from src.typing.pipeline_objects import XData
import torch
from src.framework.logging import Logger

logger = Logger()

class MinigridScorer(TransformationBlock):
    """Score the predictions of the model."""

    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        self.conversion = CONVERT[info['train']['dataset']]
        return info

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        
        data = self.conversion(data, self.info)
        
        if data.train_predictions is not None and data.train_targets is not None:
            self.calc_accuracy(data.train_predictions, data.train_targets, "Train")
        
        if data.validation_predictions is not None and data.validation_targets is not None:
            self.calc_accuracy(data.validation_predictions, data.validation_targets, "Validation")

        return data

    def calc_accuracy(self, predictions, targets, index_pretty_name: str):
        """Calculate the accuracy of the model.

        :param index_name: The name of the indice.
        :param index_pretty_name: The pretty name of the indice.
        """
                
        ti = self.info['token_index']
        ti.discrete = True  # ToDo: fetch from info, supplied by model setup

        obs_pred, reward_pred = predictions
        obs_target, reward_target = targets
        
        # Check Observation object accuaracy
        obs_pred_obj = obs_pred[..., ti.observation[0]].argmax(axis=-1)
        obs_target_obj = obs_target[..., ti.observation[0]].argmax(axis=-1)
        obs_obj_accuracy = (obs_pred_obj == obs_target_obj).sum()/obs_pred_obj.numel()
        
        obs_obj_accuracy_no_walls = (obs_pred_obj[:, 1:-1, 1:-1] == obs_target_obj[:, 1:-1, 1:-1]).sum()/obs_pred_obj[:, 1:-1, 1:-1].numel()

        # Check Observation color accuracy
        obs_pred_color = obs_pred[..., ti.observation[1]].argmax(axis=-1)
        obs_target_color = obs_target[..., ti.observation[1]].argmax(axis=-1)
        obs_color_accuracy = (obs_pred_color == obs_target_color).sum()/obs_pred_color.numel()

        # Check Observation state accuracy
        obs_pred_state = obs_pred[..., ti.observation[2]].argmax(axis=-1)
        obs_target_state = obs_target[..., ti.observation[2]].argmax(axis=-1)
        obs_state_accuracy = (obs_pred_state == obs_target_state).sum()/obs_pred_state.numel()

        # Check Observation agent accuracy
        obs_pred_agent = obs_pred[..., ti.observation[3]].argmax(axis=-1)
        obs_target_agent = obs_target[..., ti.observation[3]].argmax(axis=-1)
        obs_agent_accuracy = (obs_pred_agent == obs_target_agent).sum()/obs_pred_agent.numel()

        # Check Reward accuracy
        reward_accuracy = torch.isclose(reward_target, reward_pred, atol=0.1).sum()/len(reward_target)
        
        # Overall accuracy
        accuracy = (obs_obj_accuracy + obs_color_accuracy + obs_state_accuracy + obs_agent_accuracy + reward_accuracy) / 5
        
        # Log the results
        logger.info(f"{index_pretty_name}: Accuracy: {accuracy}")
        logger.info(f"{index_pretty_name}: Observation Object Accuracy: {obs_obj_accuracy}")
        logger.info(f"{index_pretty_name}: Observation Object Accuracy (no walls): {obs_obj_accuracy_no_walls}")
        logger.info(f"{index_pretty_name}: Observation Color Accuracy: {obs_color_accuracy}")
        logger.info(f"{index_pretty_name}: Observation State Accuracy: {obs_state_accuracy}")
        logger.info(f"{index_pretty_name}: Observation Agent Accuracy: {obs_agent_accuracy}")
        logger.info(f"{index_pretty_name}: Reward Accuracy: {reward_accuracy}")
        
        logger.log_to_external(
            message={
                f"{index_pretty_name}/Accuracy": accuracy,
                f"{index_pretty_name}/Observation Object Accuracy": obs_obj_accuracy,
                f"{index_pretty_name}/Observation Object Accuracy (no walls)": obs_obj_accuracy_no_walls,
                f"{index_pretty_name}/Observation Color Accuracy": obs_color_accuracy,
                f"{index_pretty_name}/Observation State Accuracy": obs_state_accuracy,
                f"{index_pretty_name}/Observation Agent Accuracy": obs_agent_accuracy,
                f"{index_pretty_name}/Reward Accuracy": reward_accuracy,
            },
        )
        
def convert_token_dataset(data: XData, info: dict) -> Any:
    
    def convert_tokens(data: torch.Tensor, ti: TokenIndex, obs_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.stack(data).reshape(-1, *obs_shape, data[0].shape[-1])
        obs = data[..., :-1, ti.observation_]
        obs = data.reshape(-1, *obs_shape, obs.shape[-1])
        reward = data[..., -1, ti.reward_]
        return obs, reward
    
    ti = info['token_index']
    shape = info['env_build']['observation_space'].shape[:2]
    
    if data.train_predictions is not None:
        data.train_predictions = convert_tokens(data.train_predictions, ti, shape)
    
    if data.train_targets is not None:
        data.train_targets = convert_tokens(data.train_targets, ti, shape)
        
    if data.validation_predictions is not None:
        data.validation_predictions = convert_tokens(data.validation_predictions, ti, shape)
    
    if data.validation_targets is not None:
        data.validation_targets = convert_tokens(data.validation_targets, ti, shape)
    
    return data

def convert_two_d_dataset(data: XData, info: dict) -> Any:
    return data

CONVERT = {
    "TokenDataset": convert_token_dataset,
    "TwoDDataset": convert_two_d_dataset
}