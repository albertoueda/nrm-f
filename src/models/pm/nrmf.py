import itertools
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.data.pm19.preprocessors import PM19DataProcessor

from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]

class PMBaseModel(BaseModel):

    def __init__(self, data_processor: PM19DataProcessor):

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

        super().__init__(data_processor)

    def new_input(self, name, size):
        return tf.keras.Input(shape=(size,), name=name)
        
class Naive(PMBaseModel):
    @property
    def name(self) -> str:
        return 'naive'

    def build(self):

        word_embedding = layers.Embedding(self.total_words, 
                                          self.embedding_dim, 
                                          name='word_embedding')

                                          
        texts = [word_embedding(text_input) for text_input in text_inputs]
        texts = [layers.GlobalMaxPooling1D()(text) for text in texts]

        features = texts

        x = layers.concatenate(features)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)


class NRMFSimpleQuery(PMBaseModel):
    @property
    def name(self):
        return 'nrmf_simple_query'

    def build(self):
        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')

        embeds = []
        for field in self.fields:
            field_embed = layers.GlobalMaxPooling1D()(word_embedding( self.inputs[field] ))
            embeds.append(field_embed)

        interactions = []
        query = layers.GlobalMaxPooling1D()(word_embedding( self.inputs['query'] ))
        for field_embed in embeds:
            interactions.append(tf.multiply(query, field_embed))

        x = layers.concatenate(interactions)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        # TODO test if order matters
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
