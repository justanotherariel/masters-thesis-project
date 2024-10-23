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
    data: XData = builder.transform(XData())

    num_samples = 10
    num_samples_per_env = 5
    perc_train = 0.8

    sampler = GymnasiumSampler(num_samples=num_samples, num_samples_per_env=num_samples_per_env, perc_train=perc_train)
    data = sampler.transform(data)

    # Check all arrays have the correct length
    assert len(data.observations) >= num_samples
    assert len(data.actions) == len(data.rewards) == num_samples

    # Check shape of states
    observation_size = (data.env.unwrapped.width, data.env.unwrapped.height, 3)
    assert data.observations.shape[1:] == observation_size

    # Check shape of rewards and actions
    assert data.actions.shape[1:] == data.rewards.shape[1:] == (1,)

    # Check that exactly one agent is present on the grid for every state
    target = [10, 0]  # Agent Encoding. Last position indicates direction
    mask = np.ones(data.observations.shape[:-1], dtype=bool)
    for idx in [0, 1]:
        mask &= data.observations[..., idx] == target[idx]
    assert np.all(np.sum(mask, axis=(1, 2)))

    # Check that the direction of agent is between 0 and 4
    directions = data.observations[mask][:, -1]
    assert np.all(directions >= 0)
    assert np.all(directions < 4)

    # Check len of train/validation indices and pointers
    indices_flat = data.train_indices.reshape(-1,3)
    indices_without_padding = indices_flat[indices_flat[:, 0] != -1]
    assert len(indices_without_padding) == round(num_samples * perc_train)
    for idx in indices_without_padding:
        assert idx[0] in range(len(data.observations))
        assert idx[1] in range(len(data.observations))
    
    indices_flat = data.validation_indices.reshape(-1,3)
    indices_without_padding = indices_flat[indices_flat[:, 0] != -1]
    assert len(indices_without_padding) == round(num_samples * (1 - perc_train))
    for idx in indices_without_padding:
        assert idx[0] in range(len(data.observations))
        assert idx[1] in range(len(data.observations))