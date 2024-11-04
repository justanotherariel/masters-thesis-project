from src.framework.transforming import TransformationBlock
from src.typing.pipeline_objects import XData
from typing import Any

class MinigridScorer(TransformationBlock):
    """Score the predictions of the model."""
    
    def setup(self, info: dict[str, Any]) -> dict[str, Any]:
        """Setup the transformation block.

        :param info: The input data.
        :return: The transformed data.
        """
        self.info = info
        return info

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
                
        labels = data.validation_labels
        predictions = data.validation_predictions
        
        # Check tokentype accuracy
        
        
        # Check Observation FieldType accuaracy
        
        # Check Observation color accuracy
        
        # Check Observation state accuracy
        
        # Check Reward accuracy
        
        return data

