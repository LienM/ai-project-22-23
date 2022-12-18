from itertools import product

import networkx as nx
import numpy as np
from karateclub import DeepWalk
from tqdm import tqdm
import time

from BasilRommens.dataset import part_data_set, read_data_set


def create_article_temporal_embedding_naive(embedding_fname, week_interval=None,
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


def create_article_temporal_embedding_improved(embedding_fname,
                                               week_interval=None,
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

        # get the first transactions of the customer
        first_transaction_date = c_transactions['t_dat'].min()
        first_transactions = c_transactions[
            c_transactions['t_dat'] == c_transactions['t_dat'].min()]

        # make edges between all the transactions on the same day
        for first_article_id, second_article_id in product(
                first_transactions['article_id'].values,
                first_transactions['article_id'].values):
            if first_article_id == second_article_id:
                continue
            B.add_edge(first_article_id, second_article_id)

        # iterate over other non-first transactions
        prev_transactions = first_transactions
        for transaction_date in c_transactions['t_dat'].unique():
            if transaction_date == first_transaction_date:
                continue

            # get the transactions of the current date
            current_transactions = c_transactions[
                c_transactions['t_dat'] == transaction_date]

            # get all combinations of the first and current transactions
            for prev_article_id, cur_article_id in product(
                    prev_transactions['article_id'].values,
                    current_transactions['article_id'].values):
                if prev_article_id == cur_article_id:
                    continue
                B.add_edge(prev_article_id, cur_article_id)

            # update the first transactions
            prev_transactions = current_transactions

            # make edges between all the transactions on the same day
            for first_article_id, second_article_id in product(
                    prev_transactions['article_id'].values,
                    prev_transactions['article_id'].values):
                if first_article_id == second_article_id:
                    continue
                B.add_edge(first_article_id, second_article_id)

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
                                      walk_nr=3, walk_len=10, dim=16,
                                      ):
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own
    # articles, customers, transactions = part_data_set('01')  # own

    # own
    if week_interval is None:
        week_interval = [transactions['week'].min(), transactions['week'].max()]
    transactions = transactions[transactions['week'] >= week_interval[0]]
    transactions = transactions[transactions['week'] <= week_interval[1]]

    # customers = customers[:1_000]
    # articles = articles[:1_000]
    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    B = nx.Graph()

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


def graph_embedding_original():
    for last_week in [106, 105, 104, 103, 102]:
        for n_weeks in [40, 20, 10, 5, 3, 2, 1]:
            end_interval = last_week
            begin_interval = last_week - n_weeks
            create_customer_article_embedding(
                f'../data/embedding_week_{last_week}_nr_{n_weeks}.npy',
                week_interval=[begin_interval, end_interval],
                walk_len=10, walk_nr=3)


def graph_embedding_original_walk_nr():
    last_week = 106
    end_interval = last_week
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_nr in [5, 3, 2, 1]:
            begin_interval = end_interval - n_weeks
            create_customer_article_embedding(
                f'../data/embedding_week_{last_week}_nr_{n_weeks}_walk_nr_{walk_nr}.npy',
                week_interval=[begin_interval, end_interval],
                walk_len=10, walk_nr=walk_nr)


def graph_embedding_original_walk_len():
    last_week = 106
    end_interval = last_week
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_len in [15, 10, 8, 5, 3, 2, 1]:
            begin_interval = end_interval - n_weeks
            create_customer_article_embedding(
                f'../data/embedding_week_{last_week}_nr_{n_weeks}_walk_len_{walk_len}.npy',
                week_interval=[begin_interval, end_interval],
                walk_len=walk_len, walk_nr=5)


def graph_embedding_temporal_naive():
    for last_week in [106, 105, 104, 103, 102]:
        for n_weeks in [10, 5, 3, 2, 1]:
            end_interval = last_week
            begin_interval = end_interval - n_weeks
            create_article_temporal_embedding_naive(
                f'../data/embedding_week_{last_week}_nr_{n_weeks}_temporal.npy',
                week_interval=[begin_interval, end_interval],
                walk_len=10, walk_nr=3)


def graph_embedding_temporal_improved():
    last_week = 106
    for n_weeks in [10, 5, 3, 2, 1]:
        end_interval = last_week
        begin_interval = end_interval - n_weeks
        create_article_temporal_embedding_improved(
            f'../data/embedding_week_{last_week}_nr_{n_weeks}_temporal_improved.npy',
            week_interval=[begin_interval, end_interval],
            walk_len=10, walk_nr=3)


if __name__ == '__main__':
    # graph_embedding_original()
    # graph_embedding_temporal_naive()
    # graph_embedding_temporal_improved()
    # graph_embedding_original_walk_nr()
    graph_embedding_original_walk_len()
