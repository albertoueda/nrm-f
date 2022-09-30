import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger
from pandas import DataFrame, read_csv
from tensorflow import keras
from tqdm import tqdm

from src.config.base_configs import EvalConfig
from src.layers.bias import AddBias0
from src.layers.interaction import WeightedQueryFieldInteraction, WeightedFeatureInteraction, \
    WeightedSelectedFeatureInteraction
from src.losses import pairwise_losses
from src.metrics import metrics
from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[1]


def predict(model, dataset, data_processor, verbose=1) -> float:
    """
    Expected dataset: type(val_dataset):    <class 'list'>,
        [{
        'query': 'pasta', 
        'docs': [
            {'doc_id': 3, 'label': 1}, 
            {'doc_id': 2, 'label': 0}, 
            {'doc_id': 1, 'label': 1}
        ]
        }]
    """
    ndcg_scores = []
    for example in (tqdm(dataset) if verbose > 0 else dataset):
        rows = []
        for doc in example['docs']:
            row = {
                'query': example['query'],
                'doc_id': doc['doc_id'],
                'label': doc['label']
            }
            rows.append(row)
        df = DataFrame(rows)
        x, y = data_processor.process_batch(df)
        dataset = tf.data.Dataset.from_tensor_slices((x, {'label': y})).batch(128)
        preds = model.predict(dataset, verbose=0)
        df['pred'] = preds
        y_true = df['label'].tolist()
        y_pred = df['pred'].tolist()
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))

    ndcg_score = round(np.mean(ndcg_scores), 4)
    return ndcg_score

def predict_pm(model, df, data_processor, verbose=1) -> float:
    ndcg_scores = []
    
    # queries = read_csv(f'{project_dir}/data/raw/pm19-queries.csv', index_col='qid')
    # df = df.merge(queries, how='left', left_on='qid', right_index=True)

    qids = df.qid.unique()

    for qid in qids:
        print(f'---{qid}, {df.head(1)}')
        logger.info(f'Evaluating qid {qid}...')
        x, y = data_processor.process_batch(df[df.qid == qid])

        dataset = tf.data.Dataset.from_tensor_slices((x, {'label': y})).batch(128)
        preds = model.predict(dataset, verbose=0)
        df['pred'] = preds
        y_true = df['label'].tolist()
        y_pred = df['pred'].tolist()
        ndcg_scores.append(metrics.normalized_discount_cumulative_gain(y_true, y_pred))

    ndcg_score = round(np.mean(ndcg_scores), 4)
    return ndcg_score


def evaluate_ranking_model(config: EvalConfig, model: BaseModel = None) -> float:
    if not model:
        logger.info('Load model')
        filepath = f'{project_dir}/models/{config.model_name}.h5'
        custom_objects = {
            'cross_entropy_loss': pairwise_losses.cross_entropy_loss,
            'WeightedQueryFieldInteraction': WeightedQueryFieldInteraction,
            'WeightedFeatureInteraction': WeightedFeatureInteraction,
            'WeightedSelectedFeatureInteraction': WeightedSelectedFeatureInteraction,
            'AddBias0': AddBias0,
        }
        model = keras.models.load_model(filepath, custom_objects=custom_objects)

    logger.info('Load val dataset')
    with open(f'{project_dir}/models/{config.data_processor_filename}.pkl', 'rb') as file: #TODO model is ''
        data_processor = pickle.load(file)

    if 'cookpad' in config.dataset: #TODO
        with open(f'{project_dir}/data/processed/{config.dataset}.val.pkl', 'rb') as file: #TODO cookpad
            val_dataset = pickle.load(file)
        ndcg_score = predict(model, val_dataset, data_processor, config.verbose)
    else:
        val_dataset = read_csv(f'{project_dir}/data/raw/training-queries.csv.gz')
        ndcg_score = predict_pm(model, val_dataset, data_processor, config.verbose)
    
    logger.info(f'NDCG: {ndcg_score}')

    # logger.info(f'------ Sample: type(data_processor): {type(data_processor)}')
    # logger.info(f'------ Sample: type(val_dataset):    {type(val_dataset)},\n{val_dataset}')

    return ndcg_score
