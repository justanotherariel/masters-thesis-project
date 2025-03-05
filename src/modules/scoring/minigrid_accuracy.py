from src.framework.logging import Logger
from src.framework.transforming import TransformationBlock
from src.modules.scoring.data_transform import dataset_to_list
from src.modules.training.accuracy import BaseAccuracy
from src.modules.training.torch_trainer import append_to_dict, average_dict, log_dict
from src.typing.pipeline_objects import DatasetGroup, PipelineData, PipelineInfo

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

        for dg in data.predictions:
            if dg == DatasetGroup.ALL or dg == DatasetGroup.NONE:
                continue
            dg_name = dg.name.capitalize()

            raw_data = dataset_to_list(data, dg, discretize=True, info=self._info)
            accuracies = append_to_dict({}, self.accuracy_calc(data.predictions[dg][:2], raw_data[1], raw_data[0]))

            data.accuracies[dg] = average_dict(accuracies)

            logger.info(f"Accuracies for {dg_name} (%)")
            longest_key = max([len(key) for key in data.accuracies[dg]]) + 1
            for key, value in data.accuracies[dg].items():
                value_percent = str(value * 100)[:5]
                logger.info(f"{key:<{longest_key}}: {value_percent}")
            logger.info("")

            # Log to external logger
            log_dict(data.accuracies[dg], dg_name)

        # Optimization Metric - takes transition accuracy and training time into account
        if data.model_training_time_s > 0:
            optimization_metric = data.accuracies[DatasetGroup.TRAIN]["Transition Accuracy"]

            if optimization_metric > 0.95:
                optimization_metric += 60 / data.model_training_time_s

            logger.info(f"Optimization Metric: {optimization_metric:.4f}")
            logger.log_to_external({"Optimization Metric": optimization_metric}, commit=False)
            logger.info("")

        # Log Epoch and commit
        logger.log_to_external({"Epoch": data.model_last_epoch_recorded + 1})

        return data
