import numpy as np

from src.modules.environment.gymnasium import (
    GymnasiumBuilder,
    GymnasiumSamplerRandom,
    MinigridSamplerExtensive,
    flatten_indices,
)
from src.typing.pipeline_objects import PipelineData


def test_gymnasium_builder():
    """Test the GymnasiumBuilder class."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data = builder.transform(PipelineData())
    assert data.env is not None


def test_gymnsasium_sampler_random():
    """Test the GymnasiumSampler class."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data: PipelineData = builder.transform(PipelineData())

    num_samples = 10
    num_samples_per_env = 5
    perc_train = 0.8

    sampler = GymnasiumSamplerRandom(
        num_samples=num_samples, num_samples_per_env=num_samples_per_env, perc_train=perc_train
    )
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
    train_indices = flatten_indices(data.train_indices)
    assert len(train_indices) == round(num_samples * perc_train)
    for idx in train_indices:
        assert idx[0] in range(len(data.observations))
        assert idx[1] in range(len(data.observations))
        assert idx[2] in range(len(data.actions))

    val_indices = flatten_indices(data.validation_indices)
    assert len(val_indices) == round(num_samples * (1 - perc_train))
    for idx in val_indices:
        assert idx[0] in range(len(data.observations))
        assert idx[1] in range(len(data.observations))
        assert idx[2] in range(len(data.actions))


def test_minigrid_sampler_extensive():
    """Test the MinigridSamplerExtensive class."""
    # Initialize environment
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data: PipelineData = builder.transform(PipelineData())

    # Initialize sampler with 2 train and 1 validation environment
    train_envs = 2
    validation_envs = 1
    sampler = MinigridSamplerExtensive(train_envs=train_envs, validation_envs=validation_envs)
    data = sampler.transform(data)

    # Calculate expected number of samples per environment
    env_width = data.env.unwrapped.width
    env_height = data.env.unwrapped.height
    action_space = data.env.action_space.n.item()
    # In an empty 5x5 grid, excluding walls (so 3x3 interior), with 4 directions per position
    empty_cells = (env_width - 2) * (env_height - 2)  # Interior cells excluding walls
    samples_per_env = empty_cells * 4 * action_space  # positions * directions * actions

    # Check basic array shapes and sizes
    assert len(data.observations) > 0
    assert len(data.actions) == len(data.rewards)  # Actions and rewards should match
    assert data.actions.shape[1:] == (1,)
    assert data.rewards.shape[1:] == (1,)

    # Check observation shape
    observation_size = (env_width, env_height, 4)
    assert data.observations.shape[1:] == observation_size

    # Check agent placement and direction in observations
    mask = np.ones(data.observations.shape[:-1], dtype=bool)
    mask &= data.observations[..., 3] > 0
    mask &= data.observations[..., 3] <= 4
    assert np.all(np.sum(mask, axis=(1, 2)) == 1)

    # Check objects are within valid range
    objects = data.observations[..., 0]
    assert np.all(objects >= 0)
    assert np.all(objects < 10)

    # Check colors are within valid range
    colors = data.observations[..., 1]
    assert np.all(colors >= 0)
    assert np.all(colors < 6)

    # Check states are within valid range
    states = data.observations[..., 2]
    assert np.all(states >= 0)
    assert np.all(states < 3)

    # Check actions are within action space
    assert np.all(data.actions >= 0)
    assert np.all(data.actions < action_space)

    # Check indices
    train_indices = flatten_indices(data.train_indices)
    val_indices = flatten_indices(data.validation_indices)

    # Check that indices point to valid positions in arrays
    for indices in [train_indices, val_indices]:
        for idx in indices:
            assert idx[0] in range(len(data.observations))  # x index
            assert idx[1] in range(len(data.observations))  # y index
            assert idx[2] in range(len(data.actions))  # action/reward index

    # Verify train/validation split
    assert len(train_indices) == samples_per_env * train_envs
    assert len(val_indices) == samples_per_env * validation_envs

    # Check that each position has all possible actions
    for env_indices in [train_indices, val_indices]:
        unique_positions = set()
        direction_counts = {}
        action_counts = {}

        for idx in env_indices:
            x_obs_idx = idx[0]  # Initial observation index
            pos_key = np.where(data.observations[x_obs_idx, :, :, 3] != 0)
            pos_key = (pos_key[0].item(), pos_key[1].item())  # Agent position
            unique_positions.add(pos_key)

            if pos_key not in direction_counts:
                direction_counts[pos_key] = set()
            direction_counts[pos_key].add(data.observations[x_obs_idx, pos_key[0], pos_key[1], 3])

            if pos_key not in action_counts:
                action_counts[pos_key] = set()
            action_counts[pos_key].add(data.actions[idx[2]][0])

        # Check that number of unique positions is correct
        assert len(unique_positions) == empty_cells

        # Check that each position has all possible directions
        for pos in direction_counts:
            assert len(direction_counts[pos]) == 4

        # Check that each position has all possible actions
        for pos in action_counts:
            assert len(action_counts[pos]) == action_space

    # Test that the environment is properly closed
    assert data.env.unwrapped.spec is not None  # Environment should still be valid
    try:
        data.env.step(0)
        raise AssertionError("Environment should be closed")
    except Exception:
        pass  # Expected - environment should be closed
