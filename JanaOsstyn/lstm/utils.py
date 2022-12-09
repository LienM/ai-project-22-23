import json
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences as keras_pad_sequences

"""
Some helper functions
"""

def vocabulary_size():
    """
    Get the vocabulary size (which is written to a json file)
    :return: the vocabulary size
    """
    return json.load(open('../../data/lstm/parameters.json'))['vocabulary_size']


def max_len():
    """
    Get the length of the longest user history
    :return: the vocabulary size
    """
    return json.load(open('../../data/lstm/parameters.json'))['max_len']


def pad_sequences(filename):
    """
    Add padding to the sequences that are shorter than the maximum sequence length.
    :param filename: the name of the file
    :return: the sequences (padded)
    """
    # create sequences
    transactions = pd.read_csv(filename)
    sequences = transactions['article_id'].str.split(' ').values.tolist()

    # pad sequences
    sequences_padded = keras_pad_sequences(sequences, maxlen=max_len(), padding='pre')
    sequences_padded = np.array(sequences_padded)
    return sequences_padded
