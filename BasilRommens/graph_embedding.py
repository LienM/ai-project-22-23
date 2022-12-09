import random

import joblib
import numpy as np
import pandas as pd
import tqdm

from BasilRommens.dataset import read_data_set
from BasilRommens.embed import create_embedding
from BasilRommens.knn import get_nn_index


def write_dict(c_candidates, f_name):
    pandas_dict = {'customer_id': list(), 'prediction': list()}

    c_ids = list()
    candidate_str = list()

    # encode and decode customer ids
    customer_encoder = joblib.load('../data/customer_encoder.joblib')

    for c_id, c_cand in tqdm.tqdm(c_candidates.items()):
        c_ids.append(c_id)
        candidate_str.append(' '.join(['0' + str(cand) for cand in c_cand]))

    pandas_dict['customer_id'] = customer_encoder.inverse_transform(c_ids)
    pandas_dict['prediction'] = candidate_str

    pd.DataFrame(pandas_dict).to_csv(f_name, index=False)


def get_graph_embedding_candidates(n, embedding_fname):
    # create_embedding('../data/embedding_og.npy')
    u = get_nn_index(embedding_fname)
    embedding = np.load(embedding_fname)

    # read all the data for further id processing
    articles, customers, _ = read_data_set('feather')
    a_len = len(articles)

    # get the customer id to idx map and the reverse idx to article id map
    # latter is for embeddings as those are stored as idces
    c_map = {c_id: idx + a_len
             for idx, c_id in enumerate(customers['customer_id'].values)}
    ra_map = {idx: a_id
              for idx, a_id in enumerate(articles['article_id'].values)}

    c_candidates = dict()
    for c_id in customers['customer_id'].values:
        # get map, customer id and its embedding
        mc_id = c_map[c_id]
        c_emb = embedding[mc_id]

        # get the 10 nearest neighbors
        nns = u.get_nns_by_vector(c_emb, n)
        # map nearest neighbor idx to article id
        mapped_nn = [ra_map[idx] for idx in nns]
        c_candidates[c_id] = mapped_nn

    return c_candidates


if __name__ == '__main__':
    c_candidates = get_graph_embedding_candidates(12,
                                                  '../data/embedding_og.npy')

    # name and write the predictions
    sub_name = 'graph_embeddings'
    write_dict(c_candidates, f'../out/{sub_name}.csv.gz')
