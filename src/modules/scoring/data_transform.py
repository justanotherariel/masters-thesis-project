import torch

from src.modules.training.datasets.simple import SimpleDatasetDefault
from src.typing.pipeline_objects import DatasetGroup, PipelineData


def dataset_to_list(
    data: PipelineData, ds_group: DatasetGroup, discretize=False, info=None
) -> list[list[torch.Tensor]]:
    if discretize and not info:
        raise ValueError("Discretization requires info to be passed.")

    dataset = SimpleDatasetDefault(data, ds_group=ds_group, discretize=discretize)
    if info is not None:
        dataset.setup(info)

    res = []
    for i in dataset[0]:
        res.append([])
        for _ in i:
            res[-1].append([])

    for elem in dataset:
        for i, elem_a in enumerate(elem):
            for j, elem_b in enumerate(elem_a):
                res[i][j].append(elem_b)

    # Turn into pytorch tensors
    for i, elem_a in enumerate(res):
        for j, elem_b in enumerate(elem_a):
            res[i][j] = torch.stack(elem_b)

    return res
