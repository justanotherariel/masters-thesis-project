from math import prod

import torch

from src.modules.environment.minigrid_builder import MinigridBuilder, MinigridSamplerRandom
from src.modules.training.datasets.autoregressive_token_dataset import AutoregressiveTokenDataset, TokenType
from src.typing.pipeline_objects import PipelineData


def create_token_minigrid_dataset(
    environment: str,
    num_samples: int,
    num_samples_per_env: int,
    perc_train: int,
    indices: str,
    discretize: bool = False,
) -> AutoregressiveTokenDataset:
    """Create a token minigrid dataset."""
    builder = MinigridBuilder(environment)
    sampler = MinigridSamplerRandom(num_samples, num_samples_per_env, perc_train)

    info = builder.setup({})
    info = sampler.setup(info)

    data = builder.transform(PipelineData())
    data = sampler.transform(data)

    dataset = AutoregressiveTokenDataset(data, indices, discretize)
    dataset.setup(info)

    return dataset, info


def _test_token_minigrid_dataset():
    """Test the TokenMinigridDataset class."""

    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)
    environment_shape = (5, 5)
    dataset, info = create_token_minigrid_dataset(
        environment="MiniGrid-Empty-5x5-v0",
        num_samples=num_samples_total,
        num_samples_per_env=5,
        perc_train=0.8,
        indices="train_indices",
    )

    # Check the length of the dataset
    assert len(dataset) == num_samples_train * (prod(environment_shape) + 1)

    # Access all items once to check for errors
    padding_token_list = []
    reward_token_list = []
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        #   Twice the size of the state, + 1 for SOS, + 1 for action,
        #   + 1 for SEP (Reward is the last token, thus only ever a target)
        assert x.shape == (5 * 5 * 2 + 1 + 1 + 1, 6)
        assert y.shape == (6,)

        # Check padding
        num_padding_tokens = torch.sum(x[:, 0] == TokenType.PAD.value).item()
        assert num_padding_tokens <= 5 * 5
        padding_token_list.append(num_padding_tokens)

        # Check that there is always an action token
        assert torch.sum(x[:, 4] != 255) == 1
        assert x[x[:, 4] != 255, 4].item() in range(7)

        # Check reward tokens - no reward tokens as x
        assert torch.sum(x[:, 5] != 255) == 0

        # Check reward tokens - sometimes reward tokens as y
        if y[5] != 255:
            reward_token_list.append(y[5].item())

    # Check that for each trajectory, the padding lengths are the same
    length = {}
    for num in padding_token_list:
        if num not in length:
            length[num] = 1
        else:
            length[num] += 1
    assert all(len == 80 for len in length.values())

    # Check that for each trajectory, a reward token is present
    assert len(reward_token_list) == 80


def test_token_minigrid_dataset_with_token_discretizer():
    """Test the TokenMinigridDataset class with a TokenDiscretizer."""

    num_samples_total = 50
    perc_train = 0.8
    num_samples_train = round(num_samples_total * perc_train)
    environment_shape = (5, 5)
    dataset, info = create_token_minigrid_dataset(
        environment="MiniGrid-Empty-5x5-v0",
        num_samples=num_samples_total,
        num_samples_per_env=5,
        perc_train=perc_train,
        indices="train_indices",
        discretize=True,
    )

    ti = dataset.ti

    # Check the length of the dataset
    assert len(dataset) == num_samples_train * (prod(environment_shape) + 1)

    # Access all items once to check for errors
    padding_token_list = []
    y_reward_token_list = []
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        ti.discrete = True
        assert x.shape == (5 * 5 * 2 + 1 + 1 + 1, ti.shape)
        assert y.shape == (ti.shape,)

        # Check that everything is discretized correctly
        assert x.dtype == torch.uint8
        assert y.dtype == torch.uint8

        assert torch.all(torch.sum(x[:, ti.type_], dim=-1) == 1)
        assert torch.sum(y[ti.type_]) == 1

        assert torch.all(torch.sum(x[:, ti.observation[0]], dim=-1) == 1)
        assert torch.sum(y[ti.observation[0]]) == 1

        assert torch.all(torch.sum(x[:, ti.observation[1]], dim=-1) == 1)
        assert torch.sum(y[ti.observation[1]]) == 1

        assert torch.all(torch.sum(x[:, ti.observation[2]], dim=-1) == 1)
        assert torch.sum(y[ti.observation[2]]) == 1

        assert torch.all(torch.sum(x[:, ti.action_], dim=-1) == 1)
        assert torch.sum(y[ti.action_]) == 1

        # Reward is continuous
        assert torch.sum(x[:, ti.reward_]) == 0
        assert y[ti.reward_] >= 0  # Reward is always positive

        # Check padding length
        num_padding_tokens = torch.sum(x[:, ti.type_][:, 0] == 1).item()
        assert num_padding_tokens <= 5 * 5
        padding_token_list.append(num_padding_tokens)

        # Check action tokens - always appear in x
        assert torch.sum(x[:, ti.type_][:, TokenType.ACTION.value]) == 1

        # Check action tokens - no action tokens as y
        assert y[ti.type_][TokenType.ACTION.value] == 0

        # Check reward tokens - sometimes reward tokens as y
        if y[ti.type_][TokenType.REWARD.value] == 1:
            y_reward_token_list.append(y[ti.reward_].item())
        else:
            # Otherwise it has to be an observation token
            assert y[ti.type_][TokenType.OBSERVATION.value] == 1

    # Check that for each trajectory, the padding lengths are the same
    length = {}
    for num in padding_token_list:
        if num not in length:
            length[num] = 1
        else:
            length[num] += 1
    assert all(len == num_samples_train for len in length.values())

    # Check that for each trajectory, a reward token is present
    assert len(y_reward_token_list) == num_samples_train
