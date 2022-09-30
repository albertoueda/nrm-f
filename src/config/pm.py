from typing import Tuple

from src.config.base_configs import TrainConfig, EvalConfig
from src.data.cookpad.preprocessors import ConcatDataProcessor
from src.data.cookpad.preprocessors import DataProcessor
from src.models.pm import naive, nrmf

def nrmf_simple_query_config(dataset_id: str, epochs: int, data_processor: DataProcessor) -> Tuple[
    TrainConfig, EvalConfig]:

    if not data_processor:
        data_processor = ConcatDataProcessor()

    train_config = TrainConfig(
        dataset_id=dataset_id,
        data_processor=data_processor,
        model=nrmf.NRMFSimpleQuery,
        epochs=epochs,
        verbose=2,
    )

    eval_config = EvalConfig(
        dataset_id=dataset_id,
        model_name='nrmf_simple_query',
        verbose=0,
    )
    
    return train_config, eval_config