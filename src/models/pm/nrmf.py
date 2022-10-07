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
        
        fields = [
            'background',
            'methods',
            'results',
            'conclusions'
        ]

        inputs = {
            'query'      : self.new_input('query', size=10),
            'abstract'   : self.new_input('abstract', size=1200),
            'docno'      : self.new_input('docno', size=1)
            # 'articletitle'
        }
        for field in fields:
            inputs[field] = self.new_input(field, size=300)

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')

        embeds = []
        for field in fields:
            field_embed = layers.GlobalMaxPooling1D()(word_embedding( inputs[field] ))
            embeds.append(field_embed)

        interactions = []
        query = layers.GlobalMaxPooling1D()(word_embedding( inputs['query'] ))
        for field_embed in embeds:
            interactions.append(tf.multiply(query, field_embed))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        # TODO
        inputs = list(inputs.values())
        # input_list = [
        #     inputs['query'],
        #     inputs['abstract'],
        #     inputs['background'],
        #     inputs['methods'],
        #     inputs['results'],
        #     inputs['conclusions'],
        #     inputs['docno'],
        # ]

        return keras.Model(inputs=inputs, outputs=output, name=self.name)

    def new_input(self, name, size):
        return tf.keras.Input(shape=(size,), name=name)