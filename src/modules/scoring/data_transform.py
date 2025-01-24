import torch

from src.modules.training.datasets.two_d_dataset import TwoDDataset
from src.typing.pipeline_objects import PipelineData, DatasetGroup
from typing import Any

def dataset_to_list(data: PipelineData, ds_group: DatasetGroup) -> list[list[Any]]:
    dataset = TwoDDataset(data, ds_group=ds_group, discretize=False)
    
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
