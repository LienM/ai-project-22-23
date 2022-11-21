
import numpy as np
import pandas as pd
from sklearn import preprocessing


class MultiLabelEncoder:
    """
    Object that holds all encoders so that they can be called upon at any time.
    """

    def __init__(self):
        self.encoders = {}

    def create_encoder(self, name):
        new_encoder = preprocessing.LabelEncoder()
        self.encoders[name] = new_encoder

    def encode(self, encoder_name, label, dataframe):
        encoder = self.encoders[encoder_name]
        dataframe[label] = encoder.fit_transform(dataframe[label])
        return dataframe

    def decode(self, encoder_name, label, dataframe):
        decoder = self.encoders[encoder_name]
        dataframe[label] = decoder.inverse_transform(dataframe[label])
        return dataframe


