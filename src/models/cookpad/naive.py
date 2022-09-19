from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base_model import BaseModel

project_dir = Path(__file__).resolve().parents[3]


class Naive(BaseModel):
    @property
    def name(self) -> str:
        return 'naive'

    def build(self):
        text_inputs = [
            self.new_query_input(),
            self.new_title_input(),
            self.new_ingredients_input(),
            self.new_description_input(),
        ]
        country_input = self.new_country_input()
        doc_id_input = self.new_doc_id_input()
        inputs = text_inputs + [country_input, doc_id_input]

        word_embedding = layers.Embedding(self.total_words, self.embedding_dim, name='word_embedding')
        texts = [word_embedding(text_input) for text_input in text_inputs]
        texts = [layers.GlobalMaxPooling1D()(text) for text in texts]
        country = layers.Embedding(self.total_countries, self.embedding_dim)(country_input)
        country = tf.reshape(country, shape=(-1, self.embedding_dim,))
        features = texts + [country]

        x = layers.concatenate(features)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='label')(x)

        return keras.Model(inputs=inputs, outputs=output, name=self.name)
