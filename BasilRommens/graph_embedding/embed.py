import time
from itertools import product

import networkx as nx
import numpy as np
from karateclub import DeepWalk
from tqdm import tqdm

from BasilRommens.helper.dataset import read_data_set


def connect_all_left_to_right(lnodes, rnodes, graph):
    """
    connect all the left nodes to all the right nodes
    :param lnodes: the left nodes
    :param rnodes: the right nodes
    :param graph: the graph to add the edges to
    :return: the graph with the added edges
    """
    for lnode, rnode in product(lnodes, rnodes):
        # only connect a node if the nodes aren't the same as buying twice the
        # same is assumed to be impossible
        if lnode == rnode:
            continue
        graph.add_edge(lnode, rnode)
    return graph


def create_article_temporal_embedding_naive(embedding_fname, week_interval=None,
                                            walk_nr=3, walk_len=10, dim=16):
    """
    Connect all items that are bought by a customer in a time sequence.
    :param embedding_fname: the file name to store the embedding
    :param week_interval: the interval of the weeks to use
    :param walk_nr: the # of walks to perform
    :param walk_len: the length of the walks
    :param dim: the dimension of the embedding
    :return: Nothing
    """
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own

    # own: fetches only transactions in the given week interval
    if week_interval is not None:
        transactions = transactions[transactions['week'] >= week_interval[0]]
        transactions = transactions[transactions['week'] <= week_interval[1]]

    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    B = nx.DiGraph()  # directed graph for encoding temporal information

    a_len = len(articles)

    # create a map for all the article ids
    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    # map the article ids in the transactions to the new ids
    transactions['article_id'] = transactions['article_id'].map(a_map)

    # add all the nodes to the graph
    B.add_nodes_from(range(a_len))

    # own: construct per customer a buy sequence of articles
    transactions = transactions.sort_values(by='t_dat', ascending=True)
    for c_id in tqdm(transactions['customer_id'].unique()):
        c_transactions = transactions[transactions['customer_id'] == c_id]

        # combine the article ids based on the time sequence by staggering
        B.add_edges_from(zip(c_transactions['article_id'].values[:-1],
                             c_transactions['article_id'].values[1:]))
    print(f'done constructing graph in {time.time() - start} seconds')

    start = time.time()
    # create the model for embedding the graph
    model = DeepWalk(walk_number=walk_nr, walk_length=walk_len, dimensions=dim,
                     workers=12)
    model.fit(B)
    embedding = model.get_embedding()  # fetch the embedding
    print(f'done embedding graph in {time.time() - start} seconds')

    start = time.time()
    np.save(embedding_fname, embedding)  # store the embedding
    print(f'done storing embedding {time.time() - start} seconds')


def create_article_temporal_embedding_improved(embedding_fname,
                                               week_interval=None,
                                               walk_nr=3, walk_len=10, dim=16):
    """
    create a temporal embedding by connecting all the items that occur on the
    same day. Connect the items that occur the following purchase day will be
    connected to the previous purchase day's items.
    :param embedding_fname: the file name to store the embedding
    :param week_interval: the interval of the weeks to use
    :param walk_nr: the # of walks to perform
    :param walk_len: the length of the walks
    :param dim: the dimension of the embedding
    :return: nothing
    """
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own

    # own: fetches only transactions in the given week interval
    if week_interval is not None:
        transactions = transactions[transactions['week'] >= week_interval[0]]
        transactions = transactions[transactions['week'] <= week_interval[1]]

    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    B = nx.DiGraph()  # directed graph for encoding temporal information

    a_len = len(articles)

    # create a map for all the article ids
    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    # map the article ids in the transactions to the new ids
    transactions['article_id'] = transactions['article_id'].map(a_map)

    # add all the nodes to the graph
    B.add_nodes_from(range(a_len))

    # own: construct per customer a buy sequence of articles
    transactions = transactions.sort_values(by='t_dat', ascending=True)
    for c_id in tqdm(transactions['customer_id'].unique()):
        # get all the customer transactions
        c_transactions = transactions[transactions['customer_id'] == c_id]

        # get the first transactions of the customer
        first_transaction_date = c_transactions['t_dat'].min()
        first_transactions = c_transactions[
            c_transactions['t_dat'] == c_transactions['t_dat'].min()]

        # make edges between all the transactions on the same day
        B = connect_all_left_to_right(first_transactions['article_id'].values,
                                      first_transactions['article_id'].values,
                                      B)

        # iterate over other non-first transactions on other buy days
        prev_transactions = first_transactions
        for transaction_date in c_transactions['t_dat'].unique():
            if transaction_date == first_transaction_date:
                continue

            # get the transactions of the current date
            current_transactions = c_transactions[
                c_transactions['t_dat'] == transaction_date]

            # get all combinations of the first and current transactions
            B = connect_all_left_to_right(
                prev_transactions['article_id'].values,
                current_transactions['article_id'].values, B)

            # update the first transactions
            prev_transactions = current_transactions

            # make edges between all the transactions on the same day
            B = connect_all_left_to_right(
                prev_transactions['article_id'].values,
                prev_transactions['article_id'].values, B)

        # mistakenly left this in, but shouldn't impact the results by much
        # B.add_edges_from(zip(c_transactions['article_id'].values[:-1],
        #                      c_transactions['article_id'].values[1:]))
    print(f'done constructing graph in {time.time() - start} seconds')

    start = time.time()
    # create the model for embedding the graph
    model = DeepWalk(walk_number=walk_nr, walk_length=walk_len, dimensions=dim,
                     workers=12)
    model.fit(B)
    embedding = model.get_embedding()  # fetch the embedding
    print(f'done embedding graph in {time.time() - start} seconds')

    start = time.time()
    np.save(embedding_fname, embedding)  # store the embedding
    print(f'done storing embedding {time.time() - start} seconds')


# this code is mainly based on the code of Thomas Dooms unless mentioned
# otherwise
def create_customer_article_embedding(embedding_fname, week_interval=None,
                                      walk_nr=3, walk_len=10, dim=16):
    """
    create a customer-article embedding by connecting all the items that are
    bought by a customer.
    :param embedding_fname: the file name to store the embedding
    :param week_interval: the interval of the weeks to use
    :param walk_nr: the number of walks to perform
    :param walk_len: the length of the walks
    :param dim: the dimension of the embedding
    :return: nothing
    """
    start = time.time()
    articles, customers, transactions = read_data_set('feather')  # own

    # own: determine the transactions to use in the interval
    if week_interval is not None:
        transactions = transactions[transactions['week'] >= week_interval[0]]
        transactions = transactions[transactions['week'] <= week_interval[1]]
    print(f'done reading data in {time.time() - start} seconds')

    start = time.time()
    # undirected graph for encoding customer-article interactions
    B = nx.Graph()

    a_len = len(articles)
    c_len = len(customers)

    # create a map for all the article ids and customer ids
    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    c_map = {c_id: idx + a_len
             for idx, c_id in enumerate(customers['customer_id'].values)}
    # map the article and customer ids in the transactions to their new ids
    transactions['article_id'] = transactions['article_id'].map(a_map)
    transactions['customer_id'] = transactions['customer_id'].map(c_map)

    # add the nodes of articles and customers in their own partition
    B.add_nodes_from(range(a_len), bipartite=0)
    B.add_nodes_from(range(a_len, a_len + c_len), bipartite=1)

    # for each article customer pair, add an edge between the article and
    # customer id
    B.add_edges_from(transactions[['article_id', 'customer_id']].values)
    print(f'done constructing graph in {time.time() - start} seconds')

    start = time.time()
    # create the model for embedding the graph
    model = DeepWalk(walk_number=walk_nr, walk_length=walk_len, dimensions=dim,
                     workers=12)
    model.fit(B)
    embedding = model.get_embedding()  # fetch the embedding
    print(f'done embedding graph in {time.time() - start} seconds')

    start = time.time()
    np.save(embedding_fname, embedding)  # store the embedding
    print(f'done storing embedding {time.time() - start} seconds')


def graph_embedding_original():
    """
    (correlation analysis)
    generate all the embeddings with undirected edges. We iterate over all the
    weeks and the interval of weeks to make the embedding for
    :return: nothing
    """
    for last_week in [106, 105, 104, 103, 102]:
        end_interval = last_week
        for n_weeks in [40, 20, 10, 5, 3, 2, 1]:
            begin_interval = last_week - n_weeks
            embedding_location = f'data/embedding_week_{last_week}_nr_{n_weeks}.npy'
            create_customer_article_embedding(embedding_location,
                                              week_interval=[begin_interval,
                                                             end_interval],
                                              walk_len=10, walk_nr=3)


def graph_embedding_original_walk_nr():
    """
    generate all the embeddings with undirected edges. We iterate over all the
    different interval lengths and the number of walks to make the embedding for
    :return: nothing
    """
    last_week = 106
    end_interval = last_week
    for n_weeks in tqdm([5, 3, 2, 1]):
        begin_interval = end_interval - n_weeks
        for walk_nr in [5, 3, 2, 1]:
            embedding_location = f'data/embedding_week_{last_week}_nr_{n_weeks}_walk_nr_{walk_nr}.npy'
            create_customer_article_embedding(embedding_location,
                                              week_interval=[begin_interval,
                                                             end_interval],
                                              walk_len=10, walk_nr=walk_nr)


def graph_embedding_original_walk_len():
    """
    generate all the embeddings with undirected edges. We iterate over all the
    different interval lengths and the walk length to make the embedding for
    :return: nothing
    """
    last_week = 106
    end_interval = last_week
    for n_weeks in tqdm([5, 3, 2, 1]):
        begin_interval = end_interval - n_weeks
        for walk_len in [15, 10, 8, 5, 3, 2, 1]:
            embedding_location = f'data/embedding_week_{last_week}_nr_{n_weeks}_walk_len_{walk_len}.npy'
            create_customer_article_embedding(embedding_location,
                                              week_interval=[begin_interval,
                                                             end_interval],
                                              walk_len=walk_len, walk_nr=5)


def graph_embedding_temporal_naive():
    """
    (correlation analysis)
    generate all the embeddings with the temporal naive approach, where we only
    connect for each customer one article to the next. We iterate over all the
    weeks and the interval of weeks to make the embedding for
    :return: nothing
    """
    for last_week in [106, 105, 104, 103, 102]:
        end_interval = last_week
        for n_weeks in [10, 5, 3, 2, 1]:
            begin_interval = end_interval - n_weeks
            embedding_location = f'data/embedding_week_{last_week}_nr_{n_weeks}_temporal.npy'
            create_article_temporal_embedding_naive(embedding_location,
                                                    week_interval=[
                                                        begin_interval,
                                                        end_interval],
                                                    walk_len=10, walk_nr=3)


def graph_embedding_temporal_improved():
    """
    generate all the embeddings with the temporal improved approach, where we
    connect all the items on a single day with each other and then to the next.
    We iterate over the interval of the weeks to make the embedding for
    :return: nothing
    """
    last_week = 106
    end_interval = last_week
    for n_weeks in [10, 5, 3, 2, 1]:
        begin_interval = end_interval - n_weeks
        embedding_location = f'data/embedding_week_{last_week}_nr_{n_weeks}_temporal_improved.npy'
        create_article_temporal_embedding_improved(embedding_location,
                                                   week_interval=[
                                                       begin_interval,
                                                       end_interval],
                                                   walk_len=10, walk_nr=3)


if __name__ == '__main__':
    # running the code below will generate the embeddings for the different
    # approaches with graph embeddings
    graph_embedding_original()

    graph_embedding_original_walk_nr()
    graph_embedding_original_walk_len()

    graph_embedding_temporal_naive()
    graph_embedding_temporal_improved()
