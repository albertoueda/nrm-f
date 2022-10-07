import gc
import json
import os
import sys
import warnings
from pathlib import Path
import time
from typing import Dict, Tuple

import click
import tensorflow as tf
from loguru import logger
from pandas import DataFrame
from keras.models import load_model

from src.config import config
from src.data.cloud_storage import CloudStorage
from src.evaluation import evaluate_ranking_model
from src.training import train_ranking_model

from pandas.core.common import SettingWithCopyWarning

project_dir = Path(__file__).resolve().parents[1]
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def run_experiment(dataset: str, dataset_size: int, model_name: str, epochs: int, 
                   batch_size: int, docs: Dict = None) -> Tuple[Dict, float]:
    train_config, eval_config = config.get_config(dataset, dataset_size, model_name, epochs, docs)

    logger.info(f'Training model ({model_name})...')
    model, history = train_ranking_model(train_config, batch_size)
                        
    eval_df = evaluate_ranking_model(eval_config, model, dataset_size)

    return history, eval_df

@click.command()
@click.option('--job-dir', type=str)
@click.option('--bucket-name', type=str)
@click.option('--env', type=str)
@click.option('--dataset', type=str, default='pm19')
@click.option('--dataset-size', type=str, default='sample')
@click.option('--goal', type=str, default='evaluate')
@click.option('--model-name', type=str, default='nrmf_simple_query')
@click.option('--epochs', type=int, default=1)
@click.option('--batch-size', type=int, default=2048)
def main(job_dir: str, bucket_name: str, env: str, dataset: str, dataset_size: str, goal: str,
         model_name: str, epochs: int, batch_size: int):

    # logger.add(f'{project_dir}/logs/{time.strftime("%Y-%m-%d-%Hh%M")}.log')

    if dataset not in ['cookpad', 'pm19']:
        raise ValueError(f'Unknown dataset is specified: {dataset}')

    if env == 'cloud':
        tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
        # ClusterSpec({'chief': ['cmle-training-master-e118e0f997-0:2222'], 'ps': [...], 'worker': [...]})
        cluster_info = tf_config.get('cluster', None)
        cluster_spec = tf.train.ClusterSpec(cluster_info)
        # {'type': 'worker', 'index': 3, 'cloud': 'w93a1503672d4dd09-ml'}
        task_info = tf_config.get('task', None)
        job_name, task_index = task_info['type'], task_info['index']
        logger.info(f'cluster_spec {cluster_spec}, job_name: {job_name}, task_index: {task_index}')
        logger.info(f'Task is lauched with arguments job-dir: {job_dir}, bucket-name: {bucket_name}')
    else:
        job_name = 'chief'

    if env == 'cloud':
        logger.info('Download data')
        bucket = CloudStorage(bucket_name)
        Path(f'{project_dir}/data/raw').mkdir(parents=True, exist_ok=True)
        Path(f'{project_dir}/data/processed').mkdir(parents=True, exist_ok=True)
        Path(f'{project_dir}/models').mkdir(exist_ok=True)
        filepaths = []
        # for dataset_id in dataset_ids:
        #     filepaths.append(f'data/processed/listwise.{dataset}.{dataset_id}.train.pkl')
        #     filepaths.append(f'data/processed/listwise.{dataset}.{dataset_id}.val.pkl')

        if dataset == 'cookpad':
            filepaths.append('data/raw/recipes.json')
        else:
            filepaths.append(f'data/raw/{dataset}-docs.json')
            filepaths.append(f'data/raw/{dataset}-train.csv')

        for filepath in filepaths:
            source = filepath
            destination = f'{project_dir}/{source}'
            logger.info(f'Download {source} to {destination}')
            bucket.download(source, destination)

    logger.info(f'Loading docs')
    if dataset == 'cookpad':
        from src.data.cookpad.recipes import load_raw_recipes
        docs = load_raw_recipes()
    else:
        docs = None

    logger.info(f'Run an experiment on {model_name} with dataset: {dataset}.{dataset_size}')

    if goal == 'evaluate':
        _, eval_config = config.get_config(dataset, dataset_size, model_name, epochs, docs)        
        eval_df = evaluate_ranking_model(eval_config, model=None, dataset_size=dataset_size)
    else:
        history, eval_df = run_experiment(dataset, dataset_size, model_name, epochs, batch_size, docs)
        eval_df['val_loss'] = history['val_loss'][-1]
    
    eval_df = eval_df.round(3)
    eval_df['dataset'] = dataset
    eval_df['dataset_size'] = dataset_size
    eval_df['model'] = model_name

    # VARY TRAINING SIZE
    # CHECK RUNS

    filename = f'{project_dir}/data/results/{dataset}_{dataset_size}_{model_name}_results.csv'         
    eval_df.set_index('name', drop=True).T.to_csv(filename, sep='\t')
    logger.info(f'Results:\n  {eval_df.T}')

    gc.collect()

    if env == 'cloud' and job_name == 'chief':
        for filepath in [
            f'logs/{dataset}_{model_name}_results.csv'
        ]:
            source = f'{project_dir}/{filepath}'
            destination = filepath
            logger.info(f'Upload {source} to {destination}')
            bucket.upload(source, destination)

    logger.info('Done')


if __name__ == '__main__':
    main()
