import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from src.modules.environment.gymnasium import GymnasiumBuilder, GymnasiumSamplerRandom
from src.modules.training.datasets.two_d_dataset import TwoDDataset
from src.typing.pipeline_objects import XData


def create_dataset(
    environment: str,
    num_samples: int,
    num_samples_per_env: int,
    perc_train: float,
    indices: str,
    discretize: bool = False,
) -> TwoDDataset:
    """Create a minigrid dataset preserving 2D structure."""
    builder = GymnasiumBuilder(environment)
    sampler = GymnasiumSamplerRandom(num_samples, num_samples_per_env, perc_train)

    info = builder.setup({})
    info = sampler.setup(info)

    data = builder.transform(XData())
    data = sampler.transform(data)

    dataset = TwoDDataset(data, indices, discretize)
    dataset.setup(info)

    return dataset, info


def test_two_d_dataset():
    """Test the TwoDDataset class."""
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)

    dataset, info = create_dataset(
        environment="MiniGrid-Empty-5x5-v0",
        num_samples=num_samples_total,
        num_samples_per_env=5,
        perc_train=per_train,
        indices="train_indices",
    )

    ti = dataset.ti
    environment_shape = info["env_build"]["observation_space"].shape[:-1]
    channels = len(ti.observation_)

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Access all items once to check for errors
    for i in range(len(dataset)):
        (x_obs, action), (y_obs, reward) = dataset[i]

        # Check shapes
        assert x_obs.shape == (*environment_shape, channels)
        assert y_obs.shape == (*environment_shape, channels)
        assert action.shape == (1,)
        assert reward.shape == (1,)

        # Check types
        assert x_obs.dtype == torch.uint8
        assert y_obs.dtype == torch.uint8
        assert action.dtype == torch.uint8
        assert reward.dtype == torch.float32

        # Check value ranges for MiniGrid

        ## Observation: Object
        object = x_obs[..., ti.observation[0]].squeeze()
        assert (object >= 0).all() and (object < len(OBJECT_TO_IDX)).all()
        assert (object == OBJECT_TO_IDX["goal"]).sum() == 1  # One goal present

        object = y_obs[..., ti.observation[0]].squeeze()
        assert (object >= 0).all() and (object < len(OBJECT_TO_IDX)).all()
        assert (object == OBJECT_TO_IDX["goal"]).sum() == 1  # One goal present

        ## Observation: Color
        color = x_obs[..., ti.observation[1]].squeeze()
        assert (color >= 0).all() and (color < len(COLOR_TO_IDX)).all()

        color = y_obs[..., ti.observation[1]].squeeze()
        assert (color >= 0).all() and (color < len(COLOR_TO_IDX)).all()

        ## Observation: State
        state = x_obs[..., ti.observation[2]].squeeze()
        assert (state >= 0).all() and (state < len(STATE_TO_IDX)).all()

        state = y_obs[..., ti.observation[2]].squeeze()
        assert (state >= 0).all() and (state < len(STATE_TO_IDX)).all()

        ## Observation: Agent
        agent = x_obs[..., ti.observation[3]].squeeze()
        assert (agent >= 0).all() and (agent <= 4).all()  # 4 directions + 0 (no agent)
        assert (agent != 0).sum() == 1  # One agent present

        agent = y_obs[..., ti.observation[3]].squeeze()
        assert (agent >= 0).all() and (agent <= 4).all()
        assert (agent != 0).sum() == 1

        ## Action
        assert (action >= 0).all() and (action < 7).all()  # MiniGrid has 7 actions

        ## Reward
        assert (reward >= 0).all() and (reward <= 1).all()  # MiniGrid rewards are between 0 and 1


def test_two_d_dataset_discretized():
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)

    dataset, info = create_dataset(
        environment="MiniGrid-Empty-5x5-v0",
        num_samples=num_samples_total,
        num_samples_per_env=5,
        perc_train=per_train,
        indices="train_indices",
        discretize=True,
    )

    ti = dataset.ti
    environment_shape = info["env_build"]["observation_space"].shape[:-1]

    assert len(dataset) == num_samples_train

    for i in range(len(dataset)):
        (x_obs, action), (y_obs, reward) = dataset[i]

        assert x_obs.dtype == torch.uint8
        assert y_obs.dtype == torch.uint8
        assert action.dtype == torch.uint8
        assert reward.dtype == torch.float32

        # Convert one-hot back to indices for checking
        ## Observation: Object
        object_x = torch.argmax(x_obs[..., ti.observation[0]], dim=-1)
        assert (object_x >= 0).all() and (object_x < len(OBJECT_TO_IDX)).all()
        assert (object_x == OBJECT_TO_IDX["goal"]).sum() == 1
        assert torch.sum(x_obs[..., ti.observation[0]], dim=-1).all() == 1

        object_y = torch.argmax(y_obs[..., ti.observation[0]], dim=-1)
        assert (object_y >= 0).all() and (object_y < len(OBJECT_TO_IDX)).all()
        assert (object_y == OBJECT_TO_IDX["goal"]).sum() == 1
        assert torch.sum(y_obs[..., ti.observation[0]], dim=-1).all() == 1

        ## Observation: Color
        color_x = torch.argmax(x_obs[..., ti.observation[1]], dim=-1)
        assert (color_x >= 0).all() and (color_x < len(COLOR_TO_IDX)).all()
        assert torch.sum(x_obs[..., ti.observation[1]], dim=-1).all() == 1

        color_y = torch.argmax(y_obs[..., ti.observation[1]], dim=-1)
        assert (color_y >= 0).all() and (color_y < len(COLOR_TO_IDX)).all()
        assert torch.sum(y_obs[..., ti.observation[1]], dim=-1).all() == 1

        ## Observation: State
        state_x = torch.argmax(x_obs[..., ti.observation[2]], dim=-1)
        assert (state_x >= 0).all() and (state_x < len(STATE_TO_IDX)).all()
        assert torch.sum(x_obs[..., ti.observation[2]], dim=-1).all() == 1

        state_y = torch.argmax(y_obs[..., ti.observation[2]], dim=-1)
        assert (state_y >= 0).all() and (state_y < len(STATE_TO_IDX)).all()
        assert torch.sum(y_obs[..., ti.observation[2]], dim=-1).all() == 1

        ## Observation: Agent
        agent_x = torch.argmax(x_obs[..., ti.observation[3]], dim=-1)
        assert (agent_x >= 0).all() and (agent_x <= 4).all()
        assert (agent_x != 0).sum() == 1
        assert torch.sum(x_obs[..., ti.observation[3]], dim=-1).all() == 1

        agent_y = torch.argmax(y_obs[..., ti.observation[3]], dim=-1)
        assert (agent_y >= 0).all() and (agent_y <= 4).all()
        assert (agent_y != 0).sum() == 1
        assert torch.sum(y_obs[..., ti.observation[3]], dim=-1).all() == 1

        ## Action and Reward
        action_idx = torch.argmax(action)
        assert action_idx >= 0 and action_idx < 7
        assert torch.sum(action) == 1

        assert reward >= 0 and reward <= 1


def test_collate_function():
    """Test the custom collate function."""
    batch_size = 4
    dataset, _ = create_dataset(
        environment="MiniGrid-Empty-5x5-v0",
        num_samples=100,
        num_samples_per_env=5,
        perc_train=0.8,
        indices="train_indices",
    )

    # Create a batch
    batch = [dataset[i] for i in range(batch_size)]
    (x_obs_batch, action_batch), (y_obs_batch, reward_batch) = dataset.custom_collate_fn(batch)

    # Check shapes
    assert x_obs_batch.shape == (batch_size, 5, 5, 4)
    assert y_obs_batch.shape == (batch_size, 5, 5, 4)
    assert action_batch.shape == (batch_size, 1)
    assert reward_batch.shape == (batch_size, 1)

    # Check that each item in the batch matches the original
    for i in range(batch_size):
        (x_obs, action), (y_obs, reward) = batch[i]
        assert torch.all(x_obs_batch[i] == x_obs)
        assert torch.all(action_batch[i] == action)
        assert torch.all(y_obs_batch[i] == y_obs)
        assert torch.all(reward_batch[i] == reward)
