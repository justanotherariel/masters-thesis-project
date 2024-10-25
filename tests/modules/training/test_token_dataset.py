import torch

from src.modules.environment.gymnasium import GymnasiumBuilder, GymnasiumSampler
from src.modules.training.datasets.token_dataset import TokenMinigridDataset
from src.modules.training.models.transformer import ACTION_TOKEN, PAD_TOKEN, REWARD_TOKEN
from src.typing.pipeline_objects import XData


def create_token_minigrid_dataset() -> TokenMinigridDataset:
    """Create a token minigrid dataset."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data = builder.transform(XData())

    sampler = GymnasiumSampler(num_samples=100, num_samples_per_env=5, perc_train=0.8)
    data = sampler.transform(data)

    return TokenMinigridDataset(data=data, indices="train_indices")


def test_token_minigrid_dataset():
    """Test the TokenMinigridDataset class."""
    dataset = create_token_minigrid_dataset()

    # Check the length of the dataset
    assert len(dataset) == 80 * (5 * 5 + 1)

    # Access all items once to check for errors
    padding_token_list = []
    reward_token_list = []
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        #   Twice the size of the state, + 1 for SOS, + 1 for action,
        #   + 1 for SEP (Reward is the last token, thus only ever a target)
        assert x.shape == (5 * 5 * 2 + 1 + 1 + 1, 3)
        assert y.shape == (3,)

        # Check padding
        num_padding_tokens = torch.sum((x == PAD_TOKEN).all(dim=-1)).item()
        assert num_padding_tokens <= 5 * 5
        padding_token_list.append(num_padding_tokens)

        # Check that there is always an action token
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask &= x[:, 0] == ACTION_TOKEN[0]
        mask &= x[:, 1] == ACTION_TOKEN[1]
        assert torch.sum(mask) == 1
        assert x[mask, 2].item() in range(7)

        # Check reward tokens - no reward tokens as x
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        mask &= x[:, 0] == REWARD_TOKEN[0]
        mask &= x[:, 1] == REWARD_TOKEN[1]
        assert torch.sum(mask) == 0

        # Check reward tokens - sometimes reward tokens as y
        if y[0] == REWARD_TOKEN[0] and y[1] == REWARD_TOKEN[1]:
            reward_token_list.append(y[2].item())

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
