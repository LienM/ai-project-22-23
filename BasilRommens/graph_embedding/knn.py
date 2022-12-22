import random

import numpy as np
from annoy import AnnoyIndex

from BasilRommens.helper.dataset import read_data_set


# code is based on code of Thomas Dooms unless mentioned otherwise
def get_nn_index(embedding_file):
    """
    creates an index for the embedding file and returns it
    :param embedding_file: the name of the embedding file
    :return: the index
    """
    # load the embedding
    embedding = np.load(embedding_file)  # own
    f = len(embedding[0])  # Length of item vector that will be indexed

    articles, customers, transactions = read_data_set('feather')  # own

    a_len = len(articles)  # own

    # create a new index
    t = AnnoyIndex(f, 'angular')

    # own: add all articles to the index
    for i, emb in enumerate(embedding[:a_len]):
        t.add_item(i, emb)

    # build the index
    t.build(100)  # 100 trees
    t.save('temp.ann')  # saves the index

    return t


if __name__ == '__main__':
    # test code to work with embeddings, for some embedding file with 16
    # dimensions
    u = get_nn_index('data/embedding_og.npy')
    v = [random.gauss(0, 1) for z in range(16)]
    print(u.get_nns_by_vector(v, 12))  # will find the 12 nearest neighbors
