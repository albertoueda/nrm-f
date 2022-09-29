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
        title_input = self.new_title_input()
        ingredients_input = self.new_ingredients_input()
        description_input = self.new_description_input()
        country_input = self.new_country_input()
        docno_input = self.new_docno_input()
        inputs = [query_input, title_input, ingredients_input, description_input, country_input, docno_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        query = layers.GlobalMaxPooling1D()(word_embedding(query_input))
        title = layers.GlobalMaxPooling1D()(word_embedding(title_input))
        ingredients = layers.GlobalMaxPooling1D()(word_embedding(ingredients_input))
        description = layers.GlobalMaxPooling1D()(word_embedding(description_input))
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim))

        fields = [title, ingredients, description, country]

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

    def new_title_input(self, size=20):
        return tf.keras.Input(shape=(size,), name='title')

    def new_ingredients_input(self, size=300):
        return tf.keras.Input(shape=(size,), name='ingredients')

    def new_body_input(self, size=12000):
        return tf.keras.Input(shape=(size,), name='body')