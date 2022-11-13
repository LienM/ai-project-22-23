import json
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences


def vocabulary_size():
    return json.load(open('../data/lstm/parameters.json'))['vocabulary_size']


def max_len():
    return json.load(open('../data/lstm/parameters.json'))['max_len']


def padded_sequences(filename):
    # create sequences
    # create sequences
    transactions = pd.read_csv(filename)
    sequences = transactions['article_id'].str.split(' ').values.tolist()

    # pad sequences
    sequences_padded = pad_sequences(sequences, maxlen=max_len(), padding='pre')
    sequences_padded = np.array(sequences_padded)
    return sequences_padded

