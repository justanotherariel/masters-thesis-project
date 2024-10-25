from src.modules.environment.gymnasium import GymnasiumBuilder, GymnasiumSampler
from src.modules.training.datasets.simple_dataset import SimpleMinigridDataset
from src.typing.pipeline_objects import XData


def create_simple_minigrid_dataset() -> SimpleMinigridDataset:
    """Create a simple minigrid dataset."""
    builder = GymnasiumBuilder(environment="MiniGrid-Empty-5x5-v0")
    data = builder.transform(XData())

    sampler = GymnasiumSampler(num_samples=100, num_samples_per_env=5, perc_train=0.8)
    data = sampler.transform(data)

    return SimpleMinigridDataset(data=data, indices="train_indices")


def test_simple_minigrid_dataset():
    """Test the SimpleMinigridDataset class."""
    dataset = create_simple_minigrid_dataset()

    # Check the length of the dataset
    assert len(dataset) == 80

    # Access all items once to check for errors
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Check the shape of the input and output
        assert x.shape == (5 * 5 + 1, 3)
        assert y.shape == (5 * 5 + 1, 3)
