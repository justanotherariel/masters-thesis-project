import math
from src.modules.environment.gymnasium import GymnasiumBuilder, GymnasiumSamplerRandom
from src.modules.training.datasets.simple_dataset import SimpleMinigridDataset
from src.typing.pipeline_objects import XData
from src.framework.transforming import TransformationBlock
import torch
from src.modules.training.datasets.utils import TokenType


def create_dataset(
    builder: TransformationBlock,
    sampler: TransformationBlock,
    indices: str,
    discretize: bool = False,
) -> SimpleMinigridDataset:
    """Create a token minigrid dataset."""

    info = builder.setup({})
    info = sampler.setup(info)

    data = builder.transform(XData())
    data = sampler.transform(data)

    dataset = SimpleMinigridDataset(data, indices, discretize)
    dataset.setup(info)
    
    return dataset, info


def test_with_gym_sampler_random():
    """Test the SimpleMinigridDataset class."""
    
    # Builder/Sampler Params
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)
    environment_shape = (5, 5)

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=GymnasiumSamplerRandom(num_samples=num_samples_total, num_samples_per_env=5, perc_train=per_train),
        indices="train_indices",
    )
    
    ti = dataset.ti

    # Calculate other parameters
    input_sequence_len = math.prod(environment_shape) + 1  # Observations + Action
    output_sequence_len = math.prod(environment_shape) + 1 # Observations + Reward

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Access all items once to check for errors
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        assert x.shape == (input_sequence_len, ti.shape)
        assert y.shape == (output_sequence_len, ti.shape)

        # Check all values are in the correct range
        assert x.dtype == torch.uint8
        assert y.dtype == torch.uint8
        
        assert (x[:, ti.type_] <= ti.info['type'][0][1]).all()
        assert (y[:, ti.type_] <= ti.info['type'][0][1]).all()
        
        for obs_idx in range(ti.observation_.shape[0]):
            assert (x[:, ti.observation[obs_idx]] <= ti.info['observation'][obs_idx][1]).all()
            assert (y[:, ti.observation[obs_idx]] <= ti.info['observation'][obs_idx][1]).all()
            
        for action_idx in range(ti.action_.shape[0]):
            assert (x[:, ti.action_] <= ti.info['action'][action_idx][1]).all()
            assert (y[:, ti.action_] <= ti.info['action'][action_idx][1]).all()
            
        # Check that x has an action token at the last index
        assert (x[:, ti.type_] != TokenType.ACTION.value)[:-1].all()
        assert x[-1, ti.type_] == TokenType.ACTION.value
        
        # Check that y has a reward token at the last index
        assert (y[:, ti.type_] != TokenType.REWARD.value)[:-1].all()
        assert y[-1, ti.type_] == TokenType.REWARD.value


# TODO: Implement this test
def _test_with_gym_sampler_random_discretized():
    """Test the SimpleMinigridDataset class."""
    
    # Builder/Sampler Params
    num_samples_total = 100
    per_train = 0.8
    num_samples_train = round(num_samples_total * per_train)
    environment_shape = (5, 5)

    dataset, info = create_dataset(
        builder=GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0"),
        sampler=GymnasiumSamplerRandom(num_samples=num_samples_total, num_samples_per_env=5, perc_train=per_train),
        indices="train_indices",
    )
    
    ti = dataset.ti

    # Calculate other parameters
    input_sequence_len = math.prod(environment_shape) + 1  # Observations + Action
    output_sequence_len = math.prod(environment_shape) + 1 # Observations + Reward

    # Check the length of the dataset
    assert len(dataset) == num_samples_train

    # Access all items once to check for errors
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        assert x.shape == (input_sequence_len, ti.shape)
        assert y.shape == (output_sequence_len, ti.shape)

        # Check that everything is discretized correctly
        assert x.dtype == torch.uint8
        assert y.dtype == torch.uint8
        
        assert torch.all(torch.sum(x[:, ti.type_], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.type_], dim=-1) == 1)

        assert torch.all(torch.sum(x[:, ti.observation[0]], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.observation[0]], dim=-1) == 1)

        assert torch.all(torch.sum(x[:, ti.observation[1]], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.observation[1]], dim=-1) == 1)

        assert torch.all(torch.sum(x[:, ti.observation[2]], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.observation[2]], dim=-1) == 1)
        
        assert torch.all(torch.sum(x[:, ti.observation[3]], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.observation[3]], dim=-1) == 1)

        assert torch.all(torch.sum(x[:, ti.action_], dim=-1) == 1)
        assert torch.all(torch.sum(y[:, ti.action_], dim=-1) == 1)
