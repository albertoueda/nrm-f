import abc
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
from pandas import DataFrame, read_csv
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from loguru import logger

from src.data.cookpad.preprocessors import DataProcessor
from src.dataset import load_raw_docs

project_dir = Path(__file__).resolve().parents[3]

class PM19DataProcessor(DataProcessor):
    
    #TODO link max negatives vars
    def __init__(self, docs: Dict = None, dataset_size: str = 'sample', 
                 num_words: int = 200000, max_negatives: int = 10):
        if not docs:            
            self.docs = load_raw_docs()
        else:
            self.docs = docs        
        super().__init__(self.docs, dataset_size, num_words, max_negatives)

    @property
    def doc_id_encoder(self) -> LabelEncoder:
        return self.encoder['docno']

    @property
    def total_words(self) -> int:
        return len(self.tokenizer.word_index) + 1

    @property
    def total_authors(self) -> int:
        pass

    @property
    def total_countries(self) -> int:
        pass

    def listwise_to_pairs(self, listwise_filename: str) -> DataFrame:
        # ignores the filename and return the pm csv
        #TODO gz
        self.dataset_size = 'sample'
        return read_csv(f'{project_dir}/data/raw/pm19-train-' + self.dataset_size + '.csv.gz')

    def process_df(self, df: DataFrame) -> None:
        logger.info(f'df to process (df.shape, df.columns): {df.shape}, {df.columns} ')
        logger.info(f'docs loaded: { len(self.docs) }')

        df['docno'] = df['docno'].astype(np.int64)
        df['background'] = df['docno'].apply(lambda docno: self.docs[docno]['nlmcategorybackground']).astype(str)
        df['conclusions'] = df['docno'].apply(lambda docno: self.docs[docno]['nlmcategoryconclusions']).astype(str)
        df['label'] = df['label'].astype(float)

        queries = read_csv('../data/raw/pm19-queries.csv')
        df = df.merge(queries, on='qid', how='left')
        df['query'] = df['query'].astype(str)

        logger.info(f'Processed df (shape, head): {df.shape}\n{df.head(1)} ') #TODO why log twice?
        return df

    def fit(self, df: DataFrame) -> None:
        df = self.process_df(df)

        self.encoder['docno'] = LabelEncoder()
        docnos = [docno for docno in self.docs] + [-1]
        self.encoder['docno'].fit(docnos)

        sentences = set()
        sentences |= set(df['query'])
        sentences |= set(df['background'])
        sentences |= set(df['conclusions'])

        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = self.process_df(df)

        df['docno'] = df['docno'].apply(lambda c: c if c in self.encoder['docno'].classes_ else -1)
        df['docno'] = self.encoder['docno'].transform(df['docno'])
        docno = df['docno'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['background'] = self.tokenizer.texts_to_sequences(df['background'].tolist())
        df['conclusions'] = self.tokenizer.texts_to_sequences(df['conclusions'].tolist())

        query = df['query'].tolist()
        query = pad_sequences(
            query,
            padding='post',
            truncating='post',
            maxlen=10   #TODO map direct to the model input spec
        )
        background = df['background'].tolist()
        background = pad_sequences(
            background,
            padding='post',
            truncating='post',
            maxlen=300
        )
        conclusions = df['conclusions'].tolist()
        conclusions = pad_sequences(
            conclusions,
            padding='post',
            truncating='post',
            maxlen=300
        )
        
        label = df['label'].to_numpy()

        return {
            'docno': docno,
            'query': query,
            'background': background,
            'conclusions': conclusions,
        }, label