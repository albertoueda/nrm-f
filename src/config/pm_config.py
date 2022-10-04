from typing import Tuple

from src.config.base_configs import TrainConfig, EvalConfig
from src.data.pm19.preprocessors import PM19DataProcessor
from src.models.pm import naive, nrmf

def nrmf_simple_query_config(dataset_id: str, dataset_size: str, epochs: int, 
                             data_processor: PM19DataProcessor) -> Tuple[TrainConfig, EvalConfig]:

    if not data_processor:
        data_processor = PM19DataProcessor(dataset_size=dataset_size)

    train_config = TrainConfig(
        dataset_id = dataset_id,
        data_processor = data_processor,
        data_processor_filename = 'pm19DataProcessor',
        model = nrmf.NRMFSimpleQuery,
        epochs = epochs,
        verbose = 2,
    )

    eval_config = EvalConfig(
        dataset_id = dataset_id,
        data_processor_filename = 'pm19DataProcessor',
        model_name = 'nrmf_simple_query',
        verbose = 0,
    )
    
    return train_config, eval_config