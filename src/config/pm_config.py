from typing import Tuple

from src.config.base_configs import TrainConfig, EvalConfig
from src.data.pm19.preprocessors import PM19DataProcessor
from src.models.pm import nrmf

def nrmf_simple_query_config(dataset_id: str, dataset_size: str, epochs: int, 
                             data_processor: PM19DataProcessor) -> Tuple[TrainConfig, EvalConfig]:

    if not data_processor:
        data_processor = PM19DataProcessor(dataset_size=dataset_size)

    train_config = TrainConfig(
        dataset_id = dataset_id,
        data_processor = data_processor,
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        model = nrmf.NRMFSimpleQuery,
        epochs = epochs,
        verbose = 1,
    )

    eval_config = EvalConfig(
        dataset_id = dataset_id,
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        model_name = 'nrmf_simple_query',
        verbose = 1,
    )
    
def naive_config(dataset_id: str, dataset_size: str, 
                 epochs: int, data_processor: PM19DataProcessor) -> Tuple[TrainConfig, EvalConfig]:
   
    if not data_processor:
        data_processor = PM19DataProcessor(dataset_size=dataset_size)

    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        model=nrmf.Naive,
        epochs=epochs,
        verbose=1,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='naive',
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        verbose=1,
    )
    return train_config, eval_config

def nrmf_simple_all_config(dataset_id: str, dataset_size: str,
                           epochs: int, data_processor) -> Tuple[TrainConfig, EvalConfig]:

    if not data_processor:
        data_processor = PM19DataProcessor(dataset_size=dataset_size)

    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        model=nrmf.NRMFSimpleAll,
        epochs=epochs,
        verbose=1,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='nrmf_simple_all',
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        verbose=1,
    )
    return train_config, eval_config


def nrmf_simple_query_with_1st_config(dataset_id: str, dataset_size: str, 
                                      epochs: int, data_processor) -> Tuple[TrainConfig, EvalConfig]:

    if not data_processor:
        data_processor = PM19DataProcessor(dataset_size=dataset_size)

    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        model=nrmf.NRMFSimpleQueryWith1st,
        epochs=epochs,
        verbose=1,
    )
    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='nrmf_simple_query_with_1st',
        data_processor_filename = f'pm19{dataset_size}DataProcessor',
        verbose=1,
    )
    return train_config, eval_config