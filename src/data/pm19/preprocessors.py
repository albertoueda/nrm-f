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

from src.data.cookpad.preprocessors import DataProcessor
from src.data.pm19.docs import load_raw_docs

project_dir = Path(__file__).resolve().parents[3]

class PM19DataProcessor(DataProcessor):
    
    def __init__(self, docs: Dict = None, dataset_size: str = None, num_words: int = 200000,
                 max_negatives: int = 10):
                 
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
        return read_csv(f'{project_dir}/data/raw/training-queries.csv.gz')


    def process_df(self, df: DataFrame) -> None:
        df['docno'] = df['docno'].astype(np.int64)
        df['qid'] = df['qid'].astype(str)  #TODO get query
        df['nlmcategorybackground'] = df['docno'].apply(lambda docno: self.docs[docno]['nlmcategorybackground']).astype(str)
        df['nlmcategoryconclusions'] = df['docno'].apply(lambda docno: self.docs[docno]['nlmcategoryconclusions'])
        df['label'] = df['label'].astype(float)

    def fit(self, df: DataFrame) -> None:
        self.process_df(df)

        self.encoder['docno'] = LabelEncoder()
        docnos = [docno for docno in self.docs] + [-1]
        self.encoder['docno'].fit(docnos)

        sentences = set()
        sentences |= set(df['qid']) # query
        sentences |= set(df['nlmcategorybackground'])
        sentences |= set(df['nlmcategoryconclusions'])

        self.tokenizer = Tokenizer(
            oov_token='<OOV>',
            char_level=False,
            num_words=self.num_words,
        )
        self.tokenizer.fit_on_texts(sentences)
        del sentences

    def process_batch(self, df: DataFrame) -> Tuple[Dict, List[int]]:
        df = df.copy()
        self.process_df(df)

        df['docno'] = df['docno'].apply(lambda c: c if c in self.encoder['docno'].classes_ else -1)
        df['docno'] = self.encoder['docno'].transform(df['docno'])
        docno = df['docno'].to_numpy()

        df['query'] = self.tokenizer.texts_to_sequences(df['query'].tolist())
        df['title'] = self.tokenizer.texts_to_sequences(df['title'].tolist())
        df['ingredients'] = self.tokenizer.texts_to_sequences(df['ingredients'].tolist())
        df['description'] = self.tokenizer.texts_to_sequences(df['description'].tolist())

        df['author'] = df['author'].apply(lambda c: c if c in self.encoder['author'].classes_ else '')
        df['author'] = self.encoder['author'].transform(df['author'])

        df['country'] = df['country'].apply(lambda c: c if c in self.encoder['country'].classes_ else '')
        df['country'] = self.encoder['country'].transform(df['country'])

        query = df['query'].tolist()
        query = pad_sequences(
            query,
            padding='post',
            truncating='post',
            maxlen=6
        )
        title = df['title'].tolist()
        title = pad_sequences(
            title,
            padding='post',
            truncating='post',
            maxlen=20
        )
        ingredients = df['ingredients'].tolist()
        ingredients = pad_sequences(
            ingredients,
            padding='post',
            truncating='post',
            maxlen=300
        )
        description = df['description'].tolist()
        description = pad_sequences(
            description,
            padding='post',
            truncating='post',
            maxlen=100
        )
        author = df['author'].to_numpy()
        country = df['country'].to_numpy()
        label = df['label'].to_numpy()

        return {
            'docno': docno,
            'query': query,
            'title': title,
            'ingredients': ingredients,
            'description': description,
            'author': author,
            'country': country
        }, label