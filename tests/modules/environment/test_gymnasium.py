import numpy as np

from src.modules.environment.gymnasium import GymnasiumBuilder, GymnasiumSampler
from src.typing.pipeline_objects import XData


def test_gymnasium_builder():
    """Test the GymnasiumBuilder class."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data = builder.transform(XData())
    assert data.env is not None
    
def test_gymnsasium_sampler():
    """Test the GymnasiumSampler class."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data = builder.transform(XData())
    
    num_samples = 10
    num_samples_per_env = 5
    perc_train = 0.8
    
    sampler = GymnasiumSampler(num_samples=num_samples, num_samples_per_env=num_samples_per_env, perc_train=perc_train)
    data = sampler.transform(data)
    
    # Check all arrays have the correct length
    assert len(data.x_states) == len(data.y_states) == len(data.x_actions) == len(data.y_rewards) == num_samples
    
    # Check shape of states
    size = (data.env.unwrapped.width, data.env.unwrapped.height, 3)
    assert data.x_states.shape[1:] == data.y_states.shape[1:] == size
    
    # Check shape of rewards and actions
    assert data.x_actions.shape[1:] == data.y_rewards.shape[1:] == (1,)
    
    # Check that exactly one agent is present on the grid
    target = [10, 0]    # Agent Encoding. Last position indicates direction
    mask = np.ones((num_samples, *size[:-1]), dtype=bool)
    for idx in [0, 1]:
        mask &= (data.x_states[..., idx] == target[idx])
    assert np.all(np.sum(mask, axis=(1, 2)))
    
    # Check that the direction of agent is between 0 and 4
    directions = data.x_states[mask][:, -1]
    assert np.all(directions >= 0)
    assert np.all(directions < 4)
    
    # Check length of train/validation indices
    assert len(data.train_indices) == num_samples * perc_train
    assert len(data.validation_indices) == num_samples - len(data.train_indices)