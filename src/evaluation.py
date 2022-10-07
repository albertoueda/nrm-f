import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger
from pandas import DataFrame, read_csv, concat
from tensorflow import keras
from tqdm import tqdm

from src.config.base_configs import EvalConfig
from src.layers.bias import AddBias0
from src.layers.interaction import WeightedQueryFieldInteraction, WeightedFeatureInteraction, \
    WeightedSelectedFeatureInteraction
from src.losses import pairwise_losses
from src.metrics import metrics
from src.models.base_model import BaseModel

import pyterrier as pt
if not pt.started():
    pt.init()

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

def predict_pm(model, df:DataFrame, data_processor, verbose=1) -> float:
    if df.empty: 
        raise ValueError('Testing data is empty!')

    df_qids = []
    qids = df.qid.unique()

    for qid in qids:
        logger.info(f'Evaluating qid {qid}...')
        df_qid = df[df.qid == qid]
        x, y = data_processor.process_batch(df_qid)

        dataset = tf.data.Dataset.from_tensor_slices((x, {'label': y})).batch(128)
        df_qid['pred'] = model.predict(dataset, verbose=0)

        y_true = df_qid['label'].tolist()
        y_pred = df_qid['pred'].tolist()
        logger.info(f'y_true: {y_true[:20]}')
        logger.info(f'y_pred: {y_pred[:20]}')

        df_qids.append(df_qid)

    run = concat(df_qids)
    run = process_run(run)

    return run

# eg 1 Q0 2799006 1 1.61028065 model
def process_run(run):
    run.qid = run.qid.astype(int).astype(str)#.str[4:]
    run['x'] = 'x'
    run['score'] = run.pred
    run['name'] = 'name'

    run['rank'] = run.groupby(['qid'])['score'].rank(ascending = False, method='first').astype(int)
    run = run.sort_values(['qid', 'rank'], ascending=[True, True])
    run = run.reset_index(drop=True)

    return run[['qid', 'x', 'docno', 'rank', 'score', 'name']] 

def evaluate_ranking_model(config: EvalConfig, model: BaseModel = None, 
                           dataset_size: str = 'sample'):
    logger.info('Evaluating model...')

    if not model:
        filepath = f'{project_dir}/models/{config.dataset_id}.{config.model_name}.h5'
        logger.info(f'Loading model\n  {filepath}...')
        custom_objects = {
            'cross_entropy_loss': pairwise_losses.cross_entropy_loss,
            'WeightedQueryFieldInteraction': WeightedQueryFieldInteraction,
            'WeightedFeatureInteraction': WeightedFeatureInteraction,
            'WeightedSelectedFeatureInteraction': WeightedSelectedFeatureInteraction,
            'AddBias0': AddBias0,
        }
        model = keras.models.load_model(filepath, custom_objects=custom_objects)

    #TODO remove need of stored data processors
    filename = f'{project_dir}/models/{config.data_processor_filename}.pkl'
    with open(filename, 'rb') as file:
        data_processor = pickle.load(file)
    logger.success(f'Loaded data processor:\n  {filename}')

    filename = f'{project_dir}/data/raw/pm19-test-{dataset_size}.csv.gz'
    test_dataset = read_csv(filename)
    logger.success(f'Loaded test dataset:\n  {filename}')

    run = predict_pm(model, test_dataset, data_processor, config.verbose)

    # Saving run
    #TODO if model, config.model_name may be incorrect
    logger.info(f'Run:\n{run[:10]}')
    filename = f'{project_dir}/data/runs/{config.dataset_id}-{dataset_size}-{config.model_name}.csv.gz'
    logger.info(f'Saving run to:\n  {filename}')
    run.to_csv(filename, index=False)

    data_dir = '~/data/runs-pm17-19-gla/'
    # one = io_utils.read_df(data_dir + 'df-pm17-19-THE-ONE-III.csv.gz')
    topics = read_csv(data_dir + "topics-pm-all-list-disease-gene.txt", sep='\t', names=["qid", "query"])
    topics.qid = topics.qid.astype(str)
    topics2019 = topics[topics.qid.str.startswith("2019")]
    qrels = read_csv(data_dir + 'qrels-pm-abs-all.txt', sep=' ', 
                        names=['qid', 'x', 'docno', 'label'], dtype={'qid':str})
    qrels2019 = qrels[qrels.qid.str.startswith("2019")]
    eval_metrics = ['ndcg', 'map', 'P_10', 'Rprec', 'ndcg_cut_10', 'ndcg_cut_100', "num_rel_ret", "num_ret", 
                    'P_20', 'P_30', 'P_100', 'ndcg_cut_20', 'ndcg_cut_30', ]        
 
    # sanity check
    # run = read_csv(data_dir + 'df-pm17-19-THE-ONE-III.csv.gz', dtype={'qid':str})
    # run = run[run.qid.str.startswith("2019")].reset_index(drop=True)
    # run = run[['qid', 'docno', 'score_scibert_abstracttext_orig']]
    # run = run.rename(columns={'score_scibert_abstracttext_orig':'score'})

    eval_df = pt.Experiment([run], topics2019, qrels2019, eval_metrics, names=[config.model_name])

    return eval_df
