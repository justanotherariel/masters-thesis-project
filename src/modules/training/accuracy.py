from dataclasses import dataclass

import torch

from src.typing.pipeline_objects import PipelineInfo


class BaseAccuracy:
    def __init__(self, **kwargs):
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be instantiated.")

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be called.")

    def __call__(self, predictions: tuple[torch.Tensor, torch.Tensor], targets: tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError("BaseAccuracy is an abstract class and should not be called.")


@dataclass
class MinigridAccuracy(BaseAccuracy):
    def __init__(self, **kwargs):
        pass

    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self._into = info
        self._ti = info.model_ti
        
        self._tensor_values = [self._ti.observation[i] for i in range(len(self._ti.observation))]

    def __call__(self, predictions: tuple[torch.Tensor, torch.Tensor], targets: tuple[torch.Tensor, torch.Tensor]):
        pred_obs, pred_reward = predictions
        target_obs, target_reward = targets
        
        pred_obs_argmax = torch.empty(*target_obs.shape[:3], len(self._ti.observation), dtype=torch.uint8)
        target_obs_argmax = torch.empty(*target_obs.shape[:3], len(self._ti.observation), dtype=torch.uint8)
        for idx, values in enumerate(self._tensor_values):
            pred_obs_argmax[..., idx] = pred_obs[..., values].argmax(dim=-1)
            target_obs_argmax[..., idx] = target_obs[..., values].argmax(dim=-1)
        
        object_acc = self._calc_object_color_acc(pred_obs_argmax, target_obs_argmax)
        color_acc = self._calc_color_acc(pred_obs_argmax, target_obs_argmax)
        state_acc = self._calc_state_acc(pred_obs_argmax, target_obs_argmax)
        
        one_agent_samples_acc, agent_acc = self._calc_agent_acc(pred_obs_argmax, target_obs_argmax)
        
        return {
            "Object Accuracy": object_acc,
            "Color Accuracy": color_acc,
            "State Accuracy": state_acc,
            "One Agent Samples Accuracy": one_agent_samples_acc,
            "Agent Accuracy": agent_acc
        }

    def _calc_object_color_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''Calculate the accuracy of the object component of the observation tensor. Each object class is weighted equally.'''
        
        target_class_counts = torch.bincount(targets[..., 0].view(-1), minlength=self._ti.info['observation'][0][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = targets[..., 0] == predictions[..., 0]
        pred_class_counts = torch.bincount(targets[..., 0][truth_mask], minlength=self._ti.info['observation'][0][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]
        
        return (pred_class_counts / target_class_counts).mean().item()
            
    def _calc_color_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''Calculate the accuracy of the color component of the observation tensor. Each state class is weighted equally, but is only counted if the object component on that field was correctly predicted.'''
        object_truth_mask = targets[..., 0] == predictions[..., 0]
        predictions = predictions[object_truth_mask]
        targets = targets[object_truth_mask]
        
        target_class_counts = torch.bincount(targets[..., 1].view(-1), minlength=self._ti.info['observation'][1][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]
        
        truth_mask = targets[..., 1] == predictions[..., 1]
        pred_class_counts = torch.bincount(targets[..., 1][truth_mask], minlength=self._ti.info['observation'][1][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]
        
        return (pred_class_counts / target_class_counts).mean().item()
    
    def _calc_state_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''Calculate the accuracy of the state component of the observation tensor. Each state class is weighted equally, but is only counted if the object component on that field was correctly predicted.'''
        object_truth_mask = targets[..., 0] == predictions[..., 0]
        predictions = predictions[object_truth_mask]
        targets = targets[object_truth_mask]
        
        target_class_counts = torch.bincount(targets[..., 2].view(-1), minlength=self._ti.info['observation'][2][1])
        zero_class_counts = target_class_counts == 0
        target_class_counts = target_class_counts[~zero_class_counts]

        truth_mask = targets[..., 2] == predictions[..., 2]
        pred_class_counts = torch.bincount(targets[..., 2][truth_mask], minlength=self._ti.info['observation'][2][1])
        pred_class_counts = pred_class_counts[~zero_class_counts]
        
        return (pred_class_counts / target_class_counts).mean().item()

    def _calc_agent_acc(self, predictions: torch.Tensor, targets: torch.Tensor):
        # How often was only one agent in the observation tensor?
        one_agent_samples = ((predictions[..., 3].view(-1, predictions.shape[1]*predictions.shape[2]) != 0).sum(dim=1) == 1)
        one_agent_samples_acc = one_agent_samples.sum().item() / targets.shape[0]
        
        # For the samples with only one agent, how often was the agent correctly predicted?
        predictions_tmp = predictions[one_agent_samples, :, :, 3]
        targets_tmp = targets[one_agent_samples, :, :, 3]
        agent_location = targets_tmp != 0
        agent_acc = predictions_tmp[agent_location] == targets_tmp[agent_location]
        agent_acc = agent_acc.sum().item() / targets.shape[0]
        
        # Find instances where the agent was supposed to move
        
        # Find instances where the agent was supposed to rotate
        
        # Find instances where the agent was supposed to stay
        
        return one_agent_samples_acc, agent_acc