import numpy as np
from annoy import AnnoyIndex
import random

from BasilRommens.dataset import read_data_set


# code is based on code of Thomas Dooms unless mentioned otherwise
def get_nn_index(embedding_file):
    embedding = np.load(embedding_file)  # own
    # embedding = [[2, 2, 23, 3], [1, 21, 6, 6], [32, 7, 23, 9], [2, 8, 22, 3]]
    f = len(embedding[0])  # Length of item vector that will be indexed
    articles, customers, transactions = read_data_set('feather')  # own
    a_len = len(articles)  # own

    t = AnnoyIndex(f, 'angular')
    # own
    for i, emb in enumerate(embedding[:a_len]):
        t.add_item(i, emb)

    t.build(100)  # 100 trees
    t.save('temp.ann')

    u = AnnoyIndex(f, 'angular')
    u.load('temp.ann')  # superfast, will just mmap the file

    return u


if __name__ == '__main__':
    u = get_nn_index('../data/embedding_og.npy')
    v = [random.gauss(0, 1) for z in range(16)]
    print(u.get_nns_by_vector(v, 12))  # will find the 12 nearest neighbors
