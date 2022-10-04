from typing import Set, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import click
from tqdm import tqdm
from loguru import logger

# import sys
# /home/u/git/phd-alberto-ueda/src
# sys.path.append('/home/u/git/phd-alberto-ueda/src') 
# from utils import io_utils

def build_docs(df: pd.DataFrame, dataset_size:str = 'sample'):
    df = df.drop_duplicates('docno')
    df = df.drop(columns=['qid', 'query'])

    df.to_csv('../data/raw/pm19-docs-' + dataset_size + '.csv.gz', index=False)
    logger.info(f'docs: {df.head(1)}, shape: {df.shape}')

def build_training_queries(df, train_qids, phase, dataset_size, max_negatives=5):
    rows = []
 
    for qid, group in tqdm(df.groupby('qid')):
        if qid not in train_qids:
            continue 

        logger.info(f'Processing query {qid}...')
        positives = []
        negatives = []

        for index, row in group.iterrows():
            label = 1 if row['label'] > 0 else 0
            instance = {
                'qid': row['qid'],
                'docno': row['docno'],
                'label': label
            }
            if label > 0:
                positives.append(instance)
            else:
                negatives.append(instance)

            for positive in positives:
                new_negatives = negatives[:max_negatives]
                for negative in new_negatives:
                    rows.append(positive)
                    rows.append(negative) 
                
                #TODO check possibility of empty neg examples
                negatives = negatives[max_negatives:] 
    
    df = pd.DataFrame(rows)
    logger.info(f'training queries ({phase}): shape { df.shape }\n{ df.head(2) }')
    df.to_csv(f'../data/raw/pm19-{phase}-{dataset_size}.csv.gz', index=False)

def build_test_queries(df, test_qids, dataset_size:str = 'sample'):
    df = df[['qid', 'docno', 'label']].copy()
    df = df[df.qid.isin(test_qids)]

    logger.info(f'test queries: shape { df.shape }\n{ df.head(2) }')
    df[:20].to_csv('../data/raw/pm19-test-sample.csv', index=False)
    df.to_csv('../data/raw/pm19-test-' + dataset_size + '.csv.gz', index=False)

def build_query_list(df: pd.DataFrame):
    df = df[['qid', 'query']].drop_duplicates()
    df.to_csv('../data/raw/pm19-queries.csv', index=False)
    logger.info(f'Total of queries listed: { df.shape[0] }\n{ df.head(2) }')


def load_raw_docs(dataset_size='full') -> Dict:
    docs = pd.read_csv('../data/raw/pm19-docs-' + dataset_size + '.csv.gz')
    docs = docs.drop(columns=['rank_init', 'label'])
    docs = docs.set_index('docno', drop=False)

    return docs.to_dict(orient='index')


@click.command()
@click.option('--dataset', type=str, default='pm19')
@click.option('--dataset_size', type=str, default='full')
@click.option('--source_df',   type=str, default='~/data/runs-pm17-19-gla/df-pm17-19-THE-ONE-III.csv.gz')
@click.option('--max_negatives', type=int, default=5)
@click.option('--nrows',      type=int, default=None)
def setup_training(dataset: str, dataset_size: str, source_df: str, max_negatives: int, nrows: int):
    """
    Generates the appropriate training and validation data for TREC tracks PM19 and COVID.
        $  python dataset.py
        $  python dataset.py --dataset_size=sample --nrows=20
    """

    section_cols = ['nlmcategorybackground', 'nlmcategorymethods', 'nlmcategoryresults', 'nlmcategoryconclusions']
    text_cols = ['articletitle', 'abstracttext_orig', 'abstracttext'] + section_cols
    usecols = ['qid', 'query', 'docno', 'rank_init'] + text_cols + ['label']

    if dataset == 'pm19':
        df = pd.read_csv(source_df, nrows=nrows, usecols=usecols)
        train_qids = [int(str(y) + str(i)) for y in ['2017', '2018'] for i in range(1,43)]  # 72 queries
        val_qids = [int(str(y) + str(i)) for y in ['2018'] for i in range(43,51)]           #  8 queries
        test_qids = [int(str(y) + str(i)) for y in ['2019'] for i in range(1,41)]           # 40 queries

    elif dataset == 'covid':
        pass
    else:
        raise ValueError(f"Invalid dataset type was given: {dataset}")

    # build_query_list(df)
    # build_docs(df, dataset_size)
    # build_training_queries(df, train_qids, 'train', dataset_size, max_negatives)
    # build_training_queries(df, val_qids  , 'val'  , dataset_size, max_negatives)
    # build_test_queries(df, test_qids, dataset_size)

if __name__ == '__main__':
    setup_training()