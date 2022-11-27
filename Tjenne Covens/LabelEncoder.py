
import numpy as np
import pandas as pd
from sklearn import preprocessing


class MultiLabelEncoder:
    """
    Object that holds all encoders so that they can be called upon at any time.
    """

    def __init__(self):
        self.encoders = {}
        self.decoders = {}

    def create_encoder(self, name):
        new_encoder = preprocessing.LabelEncoder()
        self.encoders[name] = new_encoder

    def encode(self, encoder_name, label, dataframe):
        # create decoder dict and store original
        decoder = pd.DataFrame(columns=['original', 'encoded'])
        decoder["original"] = dataframe[label]

        # call encoder
        encoder = self.encoders[encoder_name]
        dataframe[label] = encoder.fit_transform(dataframe[label])

        # store encoded
        decoder["encoded"] = dataframe[label]
        self.decoders[encoder_name] = decoder

        return dataframe

    def decode(self, encoder_name, label, dataframe):
        decoder = self.decoders[encoder_name]
        dataframe.rename(columns={label: 'encoded'}, inplace=True)
        dataframe = dataframe.merge(decoder, how="left", on="encoded")
        dataframe.drop(columns={"encoded"}, inplace=True)
        dataframe.rename(columns={"original": label}, inplace=True)
        return dataframe


