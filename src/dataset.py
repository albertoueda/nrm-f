import pandas as pd
import click
import json
from tqdm import tqdm
import pickle
from loguru import logger

# import sys
# /home/u/git/phd-alberto-ueda/src
# sys.path.append('/home/u/git/phd-alberto-ueda/src') 
# from utils import io_utils

def build_training_docs(df: pd.DataFrame):
    df = df.drop_duplicates('docno')
    df = df.drop(columns=['qid', 'query'])
    df = df.set_index('docno', drop=False)
    
    with open('../data/raw/docs.json', 'w') as f:
        json.dump(df.to_dict(orient='index'), f, sort_keys=True, indent=4)

    logger.info(f'docs: {df.head(1)}, shape: {df.shape}')

def build_training_queries(df):
    max_negatives = 5
    rows = []

    for qid, group in tqdm(df.groupby('qid')):
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
    logger.info(f'queries: {df.head(2)}, shape: {df.shape}')
    df.to_csv('../data/raw/training-queries.csv.gz', index=False)

    # with open(f'/home/u/git/feature-interactions-in-document-ranking/data/raw/listwise.pm19.large.train.pkl', 'wb') as file:
    #     pickle.dump(df, file)


def build_query_list(df: pd.DataFrame):
    df = df.drop_duplicates('docno')
    df = df.drop(columns=['qid', 'query'])
    df = df.set_index('docno', drop=False)
    
    with open('../data/raw/docs.json', 'w') as f:
        json.dump(df.to_dict(orient='index'), f, sort_keys=True, indent=4)
#

    logger.info(f'docs: {df.head(1)}, shape: {df.shape}')

@click.command()
@click.option('--dataset_id', type=str, default='pm19')
@click.option('--data_dir',   type=str, default='~/data/runs-pm17-19-gla/')
@click.option('--nrows',      type=int, default=20)
def setup_training(dataset_id: str, data_dir: str, nrows: int):

    section_cols = ['nlmcategorybackground', 'nlmcategorymethods', 'nlmcategoryresults', 'nlmcategoryconclusions']
    text_cols = ['articletitle', 'abstracttext_orig', 'abstracttext'] + section_cols

    df = pd.read_csv(data_dir + 'df-pm17-19-THE-ONE-III.csv.gz', nrows=nrows, 
                     usecols = ['qid', 'query', 'docno', 'rank_init'] + text_cols + ['label'])

    build_training_docs(df)
    build_training_queries(df)


if __name__ == '__main__':
    setup_training()