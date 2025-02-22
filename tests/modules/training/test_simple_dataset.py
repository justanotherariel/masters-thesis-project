import math

import torch

from src.framework.transforming import TransformationBlock
from src.modules.environment.gymnasium import GymnasiumBuilder, MinigridSamplerRandom, MinigridSamplerExtensive
from src.modules.training.datasets.token_dataset import TokenDataset
from src.modules.training.datasets.utils import TokenType
from src.typing.pipeline_objects import PipelineData


def create_dataset(
    builder: TransformationBlock,
    sampler: TransformationBlock,
    indices: str,
    discretize: bool = False,
) -> TokenDataset:
    """Create a token minigrid dataset."""

    info = builder.setup({})
    info = sampler.setup(info)

    data = builder.transform(PipelineData())
    data = sampler.transform(data)

    dataset = TokenDataset(data, indices, discretize)
    dataset.setup(info)

    return dataset, info


def general_non_dis_test(dataset, environment_shape):
    ti = dataset.ti

    # Calculate other parameters
    input_sequence_len = math.prod(environment_shape) + 1  # Observations + Action
    output_sequence_len = math.prod(environment_shape) + 1  # Observations + Reward

    # Access all items once to check for errors
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Set the discrete flag to False
        ti.discrete = False

        # Check the shape of the input and output
        assert x.shape == (input_sequence_len, ti.shape)
        assert y.shape == (output_sequence_len, ti.shape)

        # Check all values are in the correct range
        assert x.dtype == torch.uint8
        assert y.dtype == torch.uint8

        assert (x[:, ti.type_] <= ti.info["type"][0][1]).all()
        assert (y[:, ti.type_] <= ti.info["type"][0][1]).all()

        for obs_idx in range(len(ti.observation)):
            assert (x[:, ti.observation[obs_idx]] <= ti.info["observation"][obs_idx][1]).all()
            assert (y[:, ti.observation[obs_idx]] <= ti.info["observation"][obs_idx][1]).all()

        for action_idx in range(len(ti.action)):
            assert (x[:, ti.action_] <= ti.info["action"][action_idx][1]).all()
            assert (y[:, ti.action_] <= ti.info["action"][action_idx][1]).all()

        # Check that x has an action token at the last index
        assert (x[:, ti.type_] != TokenType.ACTION.value)[:-1].all()
        assert x[-1, ti.type_] == TokenType.ACTION.value

        # Check that y has a reward token at the last index
        assert (y[:, ti.type_] != TokenType.REWARD.value)[:-1].all()
        assert y[-1, ti.type_] == TokenType.REWARD.value


def test_with_gym_sampler_random():
    """Test the SimpleMinigridDataset class."""

    # Builder/Sampler Params
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)
    environment_shape = (5, 5)

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=MinigridSamplerRandom(num_samples=num_samples_total, num_samples_per_env=5, perc_train=per_train),
        indices="train_indices",
    )

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Run the general test
    general_non_dis_test(dataset, environment_shape)


def test_with_minigrid_sampler_extensive():
    """Test the SimpleMinigridDataset class."""

    # Builder/Sampler Params
    env_shape = (5, 5)
    samples_per_env = (env_shape[0] - 2) * (env_shape[1] - 2) * 4 * 7
    tain_envs = 2
    num_samples_train = samples_per_env * tain_envs

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=MinigridSamplerExtensive(train_envs=tain_envs, validation_envs=1),
        indices="train_indices",
    )

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Run the general test
    general_non_dis_test(dataset, env_shape)


def general_dis_test(dataset, environment_shape):
    ti = dataset.ti

    # Calculate other parameters
    input_sequence_len = math.prod(environment_shape) + 1  # Observations + Action
    output_sequence_len = math.prod(environment_shape) + 1  # Observations + Reward

    # Access all items once to check for errors
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Set the discrete flag to True
        ti.discrete = True

        # Check the shape of the input and output
        assert x.shape == (input_sequence_len, ti.shape)
        assert y.shape == (output_sequence_len, ti.shape)

        # Check that everything is discretized correctly
        assert x.dtype == torch.uint8
        assert y.dtype == torch.uint8

        assert torch.all(torch.sum(x[:, ti.type_], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.type_], dim=-1) == 1)

        for obs_idx in range(len(ti.observation)):
            assert torch.all(torch.sum(x[:, ti.observation[obs_idx]], dim=-1) == 1)
            assert torch.all(torch.sum(y[:, ti.observation[obs_idx]], dim=-1) == 1)

        for action_idx in range(len(ti.action)):
            assert torch.all(torch.sum(x[:, ti.action[action_idx]], dim=-1) == 1)
            assert torch.all(torch.sum(y[:, ti.action[action_idx]], dim=-1) == 1)

        # Check that x has an action token at the last index
        assert (x[:-1, ti.type_][:, TokenType.ACTION.value] != 1).all()
        assert x[-1, ti.type_][TokenType.ACTION.value] == 1

        # Check that y has a reward token at the last index
        assert (y[:-1, ti.type_][:, TokenType.REWARD.value] != 1).all()
        assert y[-1, ti.type_][TokenType.REWARD.value] == 1


def test_with_gym_sampler_random_discretized():
    """Test the SimpleMinigridDataset class."""

    # Builder/Sampler Params
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)
    environment_shape = (5, 5)

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=MinigridSamplerRandom(num_samples=num_samples_total, num_samples_per_env=5, perc_train=per_train),
        indices="train_indices",
        discretize=True,
    )

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Run the general test
    general_dis_test(dataset, environment_shape)


def test_with_minigrid_sampler_extensive_discretized():
    """Test the SimpleMinigridDataset class."""

    # Builder/Sampler Params
    env_shape = (5, 5)
    samples_per_env = (env_shape[0] - 2) * (env_shape[1] - 2) * 4 * 7
    tain_envs = 2
    num_samples_train = samples_per_env * tain_envs

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=MinigridSamplerExtensive(train_envs=tain_envs, validation_envs=1),
        indices="train_indices",
        discretize=True,
    )

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Run the general test
    general_dis_test(dataset, env_shape)
