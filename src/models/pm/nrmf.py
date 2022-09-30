import itertools
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]


class NRMFSimpleQuery(BaseModel):
    @property
    def name(self):
        return 'nrmf_simple_query'

    def build(self):
        query_input = self.new_query_input()
        background_input = self.new_field_input('background')
        conclusions_input = self.new_field_input('conclusions')
        docno_input = self.new_docno_input()
        inputs = [query_input, background_input, conclusions_input, docno_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        query       = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        background  = layers.GlobalMaxPooling1D()(word_embedding(background_input))
        conclusions = layers.GlobalMaxPooling1D()(word_embedding(conclusions_input))

        fields = [background, conclusions]

        interactions = []
        for field in fields:
            interactions.append(tf.multiply(query, field))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
        
    def new_docno_input(self):
        return tf.keras.Input(shape=(1,), name='docno')

    def new_query_input(self, size=10):
        return tf.keras.Input(shape=(size,), name='query')

    def new_field_input(self, name, size=300):
        return tf.keras.Input(shape=(size,), name=name)

    def new_body_input(self, size=12000):
        return tf.keras.Input(shape=(size,), name='body')