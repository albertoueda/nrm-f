import abc
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from loguru import logger

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

        self.fields = [
            # 'articletitle',
            'background',
            'methods',
            'results',
            'conclusions'
        ]
        self.queries = read_csv('../data/raw/pm19-queries.csv')
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

    # Overwrited to return pairs file directly
    def listwise_to_pairs(self, listwise_filename: str) -> DataFrame:

        if 'train' in listwise_filename:
            filename = f'{project_dir}/data/raw/pm19-train-' + self.dataset_size + '.csv.gz'
            logger.success(f'-------------- Read:\n{filename}')
            return read_csv(filename)
        else:
            filename = f'{project_dir}/data/raw/pm19-val-' + self.dataset_size + '.csv.gz'
            logger.success(f'-------------- Read:\n{filename}')
            return read_csv(filename)

    def process_df(self, df: DataFrame) -> None:
        # logger.info(f'df to process (df.shape, df.columns): {df.shape}, {df.columns} ')
        # logger.info(f'docs loaded: { len(self.docs) }')

        df['docno'] = df['docno'].astype(np.int64)
        df['abstract'] = df['docno'].apply(lambda docno: self.docs[docno]['abstracttext_orig']).astype(str)
        df['label'] = df['label'].astype(float)

        for field in self.fields:
            df[field] = df['docno'].apply(lambda docno: self.docs[docno][f'nlmcategory{field}']).astype(str)

        df = df.merge(self.queries, on='qid', how='left')
        df['query'] = df['query'].astype(str)

        # TODO change log level to info
        # logger.debug(f'Processed df (shape, head): {df.shape}\n{df.head(1)} ') 
        return df

    def fit(self, df: DataFrame) -> None:
        df = self.process_df(df)

        self.encoder['docno'] = LabelEncoder()
        docnos = [docno for docno in self.docs] + [-1]
        self.encoder['docno'].fit(docnos)

        sentences = set()
        sentences |= set(df['query'])
        for text_field in self.fields + ['abstract']:
            sentences |= set(df[text_field])

        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

    def get_sequence(self, df, col, maxlen):
        return pad_sequences(
            df[col].tolist(),
            padding='post',
            truncating='post',
            maxlen=maxlen   #TODO map direct to the model input spec
        )

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = self.process_df(df)

        df['docno'] = df['docno'].apply(lambda c: c if c in self.encoder['docno'].classes_ else -1)
        df['docno'] = self.encoder['docno'].transform(df['docno'])
        docno = df['docno'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        
        for text_field in self.fields + ['abstract']:
            df[text_field] = self.tokenizer.texts_to_sequences(df[text_field].tolist())

        label = df['label'].to_numpy()

        #TODO check field sizes
        dict_sequences = {
            'docno':       docno,
            'query':       self.get_sequence(df, 'query', maxlen=10),
            'abstract':    self.get_sequence(df, 'abstract', maxlen=1200),
            'background':  self.get_sequence(df, 'background', maxlen=300),
            'methods':     self.get_sequence(df, 'methods', maxlen=300),
            'results':     self.get_sequence(df, 'results', maxlen=300),
            'conclusions': self.get_sequence(df, 'conclusions', maxlen=300),
        }
        
        return dict_sequences, label