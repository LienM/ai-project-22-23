import random
import time
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from BasilRommens.dataset import read_data_set
from BasilRommens.embed import create_customer_article_embedding
from BasilRommens.knn import get_nn_index


def write_dict(c_candidates, f_name):
    pandas_dict = {'customer_id': list(), 'prediction': list()}

    c_ids = list()
    candidate_str = list()

    # encode and decode customer ids
    customer_encoder = joblib.load('../data/customer_encoder.joblib')

    for c_id, c_cand in c_candidates.items():
        c_ids.append(c_id)
        candidate_str.append(' '.join(['0' + str(cand) for cand in c_cand]))

    pandas_dict['customer_id'] = customer_encoder.inverse_transform(c_ids)
    pandas_dict['prediction'] = candidate_str

    pd.DataFrame(pandas_dict).to_csv(f_name, index=False)


def get_graph_embedding_candidates_temporal(n, embedding_fname):
    u = get_nn_index(embedding_fname)
    embedding = np.load(embedding_fname)

    # read all the data for further id processing
    articles, customers, transactions = read_data_set('feather')

    # get the customer id to idx map and the reverse idx to article id map
    # latter is for embeddings as those are stored as idces
    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    ra_map = {idx: a_id
              for idx, a_id in enumerate(articles['article_id'].values)}

    c_candidates = dict()
    transactions = transactions.set_index('customer_id')
    transactions = transactions.sort_values(by='t_dat')
    for c_id in customers['customer_id'].values:
        # get the latest item bought by customer
        try:
            a_ids = transactions.iloc[c_id]['article_id']
        except KeyError as e:
            continue

        # check type in order to get the latest id correctly
        if type(a_ids) == np.uint32:
            latest_a_id = a_map[a_ids]
        else:
            latest_a_id = a_map[a_ids.values[-1]]
        latest_a_emb = embedding[latest_a_id]

        # get n nearest neighbors
        nns = u.get_nns_by_vector(latest_a_emb, n)

        # map nearest neighbor idx to article id
        mapped_nn = map(lambda x: ra_map[x], nns)
        c_candidates[c_id] = mapped_nn

    return c_candidates


def get_graph_embedding_candidates(n, embedding_fname):
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

        # get n nearest neighbors
        nns = u.get_nns_by_vector(c_emb, n)

        # map nearest neighbor idx to article id
        mapped_nn = [ra_map[idx] for idx in nns]
        c_candidates[c_id] = mapped_nn

    return c_candidates


def graph_submission():
    for last_week in tqdm(range(106, 101, -1)):
        for n_weeks in tqdm([40, 20, 10, 5, 3, 2, 1]):
            sub_name = f'embedding_week_{last_week}_nr_{n_weeks}'
            c_candidates = get_graph_embedding_candidates(12,
                                                          f'../data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'../out/{sub_name}.csv.gz')


def graph_temporal_submission():
    for last_week in tqdm(range(106, 101, -1)):
        for n_weeks in tqdm([10, 5, 3, 2, 1]):
            sub_name = f'embedding_week_{last_week}_nr_{n_weeks}_temporal'
            c_candidates = get_graph_embedding_candidates_temporal(12,
                                                                   f'../data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'../out/{sub_name}.csv.gz')


def graph_temporal_improved_submission():
    for n_weeks in tqdm([10, 5, 3, 2, 1]):
        sub_name = f'embedding_week_106_nr_{n_weeks}_temporal_improved'
        c_candidates = get_graph_embedding_candidates_temporal(12,
                                                               f'../data/{sub_name}.npy')

        # write the predictions
        write_dict(c_candidates, f'../out/{sub_name}.csv.gz')


def graph_walk_nr_submission():
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_nr in [5, 3, 2, 1]:
            sub_name = f'embedding_week_106_nr_{n_weeks}_walk_nr_{walk_nr}'
            c_candidates = get_graph_embedding_candidates(12,
                                                          f'../data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'../out/{sub_name}.csv.gz')


def graph_walk_len_submission():
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_len in [15, 10, 8, 5, 3, 2, 1]:
            sub_name = f'embedding_week_106_nr_{n_weeks}_walk_len_{walk_len}'
            c_candidates = get_graph_embedding_candidates(12,
                                                          f'../data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'../out/{sub_name}.csv.gz')


if __name__ == '__main__':
    # graph_submission()
    # graph_temporal_submission()
    # graph_temporal_improved_submission()
    graph_walk_nr_submission()
    graph_walk_len_submission()
