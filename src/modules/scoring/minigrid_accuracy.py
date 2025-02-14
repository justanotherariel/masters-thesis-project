from src.modules.training.torch_trainer import log_dict
from src.modules.training.accuracy import BaseAccuracy
from src.typing.pipeline_objects import PipelineInfo, PipelineData, DatasetGroup
from src.modules.scoring.data_transform import dataset_to_list

from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock

logger = Logger()

class MinigridAccuracy(TransformationBlock):
    accuracy_calc: BaseAccuracy
    
    def __init__(self, accuarcy_calc: BaseAccuracy):
        self.accuracy_calc = accuarcy_calc
        
    def setup(self, info: PipelineInfo) -> PipelineInfo:
        self.accuracy_calc.setup(info)
        self._info = info
        return info
    
    def custom_transform(self, data: PipelineData) -> PipelineData:
        logger.info("Calculating accuracies")
        data.accuracies = {}
        
        for dg in data.predictions.keys():
            if dg == DatasetGroup.ALL or dg == DatasetGroup.NONE:
                continue
            dg_name = dg.name.capitalize()
            
            raw_data = dataset_to_list(data, dg, discretize=True, info=self._info)
            data.accuracies[dg] = self.accuracy_calc(data.predictions[dg], raw_data[1], raw_data[0])

            logger.info(f"Accuracies for {dg_name}")
            longest_key = max([len(key) for key in data.accuracies[dg].keys()])
            for key, value in data.accuracies[dg].items():
                logger.info(f"{key:<{longest_key}}: {value}")
            logger.info("")

            # If the predications are from a cached model, log final accuracies to wandb
            if not data.logged_accuracies_to_wandb:
                log_dict(data.accuracies[dg], -1, dg_name)
        
        return data