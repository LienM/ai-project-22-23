from itertools import product

import networkx as nx
import numpy as np
from karateclub import DeepWalk
from tqdm import tqdm
import time

from BasilRommens.dataset import part_data_set, read_data_set


def create_article_temporal_embedding(embedding_fname, week_interval=None,
                                      walk_nr=3, walk_len=10, dim=16):
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own
    # articles, _, transactions = part_data_set('01')  # own

    if week_interval is None:
        week_interval = [transactions['week'].min(), transactions['week'].max()]
    transactions = transactions[transactions['week'] >= week_interval[0]]
    transactions = transactions[transactions['week'] <= week_interval[1]]

    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    B = nx.DiGraph()

    a_len = len(articles)

    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}

    transactions['article_id'] = transactions['article_id'].map(a_map)

    B.add_nodes_from(range(a_len))

    # construct per customer a buy sequence of articles
    transactions = transactions.sort_values(by='t_dat', ascending=True)
    for c_id in tqdm(transactions['customer_id'].unique()):
        c_transactions = transactions[transactions['customer_id'] == c_id]
        B.add_edges_from(zip(c_transactions['article_id'].values[:-1],
                             c_transactions['article_id'].values[1:]))
    print(f'done constructing graph in {time.time() - start} seconds')

    start = time.time()
    model = DeepWalk(walk_number=walk_nr, walk_length=walk_len, dimensions=dim,
                     workers=12)
    model.fit(B)
    embedding = model.get_embedding()
    print(f'done embedding graph in {time.time() - start} seconds')

    start = time.time()
    np.save(embedding_fname, embedding)
    print(f'done storing embedding {time.time() - start} seconds')


# this code is mainly based on the code of Thomas Dooms unless mentioned
# otherwise
def create_customer_article_embedding(embedding_fname, week_interval=None,
                                      walk_nr=3, walk_len=10, dim=16):
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own
    # articles, customers, transactions = part_data_set('01')  # own

    if week_interval is None:
        week_interval = [transactions['week'].min(), transactions['week'].max()]
    transactions = transactions[transactions['week'] >= week_interval[0]]
    transactions = transactions[transactions['week'] <= week_interval[1]]

    # customers = customers[:1_000]
    # articles = articles[:1_000]
    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    B = nx.DiGraph()

    a_len = len(articles)
    c_len = len(customers)

    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    c_map = {c_id: idx + a_len
             for idx, c_id in enumerate(customers['customer_id'].values)}

    transactions['article_id'] = transactions['article_id'].map(a_map)
    transactions['customer_id'] = transactions['customer_id'].map(c_map)

    B.add_nodes_from(range(a_len), bipartite=0)
    B.add_nodes_from(range(a_len, a_len + c_len), bipartite=1)

    # B.add_edges_from([(x, y) for x, y in product(range(50), range(a_len + 50))])
    B.add_edges_from(transactions[['article_id', 'customer_id']].values)
    print(f'done constructing graph in {time.time() - start} seconds')

    start = time.time()
    model = DeepWalk(walk_number=walk_nr, walk_length=walk_len, dimensions=dim,
                     workers=12)
    model.fit(B)
    embedding = model.get_embedding()
    print(f'done embedding graph in {time.time() - start} seconds')

    start = time.time()
    np.save(embedding_fname, embedding)
    print(f'done storing embedding {time.time() - start} seconds')


if __name__ == '__main__':
    for last_week in [106, 105, 104, 103, 102]:
        for n_weeks in [10, 5, 3, 2, 1]:
            end_interval = last_week
            begin_interval = last_week - n_weeks
            # create_customer_article_embedding(
            #     f'../data/embedding_week_{last_week}_nr_{n_weeks}.npy',
            #     week_interval=[begin_interval, end_interval],
            #     walk_len=10, walk_nr=3)
            create_article_temporal_embedding(
                f'../data/embedding_week_{last_week}_nr_{n_weeks}_temporal.npy',
                week_interval=[begin_interval, end_interval],
                walk_len=10, walk_nr=3)
