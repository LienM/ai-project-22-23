import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from BasilRommens.helper.dataset import read_data_set
from BasilRommens.graph_embedding.knn import get_nn_index


def write_dict(c_candidates, f_name):
    """
    write a customers candidates to a csv file
    :param c_candidates: the candidates in a dict per customer
    :param f_name: the file name to write the candidates to
    :return: nothing
    """
    # the dict to use for converting to pandas df to save to csv
    pandas_dict = {'customer_id': list(), 'prediction': list()}

    # store a list of customer ids and the candidates in string form
    c_ids = list()
    candidate_str = list()

    # fetch customer id encoding
    customer_encoder = joblib.load('data/customer_encoder.joblib')

    # convert the candidates to string format per customer
    for c_id, c_cand in c_candidates.items():
        c_ids.append(c_id)
        candidate_str.append(' '.join(['0' + str(cand) for cand in c_cand]))

    # store the customer ids while transforming them and the candidates
    pandas_dict['customer_id'] = customer_encoder.inverse_transform(c_ids)
    pandas_dict['prediction'] = candidate_str

    # write the candidates to a csv file
    pd.DataFrame(pandas_dict).to_csv(f_name, index=False)


def get_graph_embedding_candidates_temporal(n_candidates, embedding_fname):
    """
    get the graph embeddings in temporal case as it slightly differs in what is
    encoded in the embeddings
    :param n_candidates: the number of candidates to return
    :param embedding_fname: the file name where the embeddings are stored
    :return: the customer id to candidates index
    """
    # get the index for the embedding
    u = get_nn_index(embedding_fname)
    # load the embedding
    embedding = np.load(embedding_fname)

    # read all the data for further id processing
    articles, customers, transactions = read_data_set('feather')

    # get the customer id to idx map and the reverse idx to article id map
    # latter is for embeddings as those are stored as idces
    a_map = {a_id: idx
             for idx, a_id in enumerate(articles['article_id'].values)}
    ra_map = {idx: a_id
              for idx, a_id in enumerate(articles['article_id'].values)}

    # get the customer id to candidate map
    c_candidates = dict()

    # set index for customer id to retrieve this faster and sort by date to
    # fetch last item
    transactions = transactions.set_index('customer_id')
    transactions = transactions.sort_values(by='t_dat')
    for c_id in customers['customer_id'].values:
        # get the latest item bought by customer
        try:
            a_ids = transactions.iloc[c_id]['article_id']
        except KeyError as e:
            continue

        # check type in order to get the last id correctly
        if type(a_ids) == np.uint32:
            latest_a_id = a_map[a_ids]
        else:
            latest_a_id = a_map[a_ids.values[-1]]
        latest_a_emb = embedding[latest_a_id]

        # get n nearest neighbors for the last embedding
        nns = u.get_nns_by_vector(latest_a_emb, n_candidates)

        # map nearest neighbor idx to article id
        mapped_nn = map(lambda x: ra_map[x], nns)
        c_candidates[c_id] = mapped_nn

    # return dict with customer id to candidates
    return c_candidates


def get_graph_embedding_candidates(n_candidates, embedding_fname):
    """
    use the graph embeddings from the undirected graph aka the non-temporal
    graph to get the candidates per customer
    :param n_candidates: the number of candidates per customer
    :param embedding_fname: the name of the file where the embeddings are stored
    :return: the customer id to candidates index
    """
    # create the index for the embeddings
    u = get_nn_index(embedding_fname)
    # load the embeddings
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
        nns = u.get_nns_by_vector(c_emb, n_candidates)

        # map nearest neighbor idx to article id
        mapped_nn = [ra_map[idx] for idx in nns]
        c_candidates[c_id] = mapped_nn

    # return dict with customer id to candidates
    return c_candidates


def graph_submission_original():
    """
    (correlation analysis)
    create the submissions for the graph embeddings that use the bipartite graph
    and undirected edges.
    :return: nothing
    """
    for last_week in tqdm(range(106, 101, -1)):
        for n_weeks in tqdm([40, 20, 10, 5, 3, 2, 1]):
            sub_name = f'embedding_week_{last_week}_nr_{n_weeks}'

            c_candidates = get_graph_embedding_candidates(12,
                                                          f'data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'out/{sub_name}.csv.gz')


def graph_submission_temporal_naive():
    """
    (correlation analysis)
    create the submissions for the graph embeddings that use the temporal graph
    in the naive way aka using only single item to item directed edges.
    :return: nothing
    """
    for last_week in tqdm(range(106, 101, -1)):
        for n_weeks in tqdm([10, 5, 3, 2, 1]):
            sub_name = f'embedding_week_{last_week}_nr_{n_weeks}_temporal'

            c_candidates = get_graph_embedding_candidates_temporal(12,
                                                                   f'data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'out/{sub_name}.csv.gz')


def graph_submission_temporal_improved():
    """
    create the submissions for the graph embeddings that use the temporal graph
    where the edges are directed and connect to all the items on the same day.
    :return: nothing
    """
    for n_weeks in tqdm([10, 5, 3, 2, 1]):
        sub_name = f'embedding_week_106_nr_{n_weeks}_temporal_improved'

        c_candidates = get_graph_embedding_candidates_temporal(12,
                                                               f'data/{sub_name}.npy')

        # write the predictions
        write_dict(c_candidates, f'out/{sub_name}.csv.gz')


def graph_submission_walk_nr():
    """
    create the submissions for the graph embeddings that use undirected edges
    from customer to item and that vary the walk number and interval duration.
    :return: nothing
    """
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_nr in [5, 3, 2, 1]:
            sub_name = f'embedding_week_106_nr_{n_weeks}_walk_nr_{walk_nr}'

            c_candidates = get_graph_embedding_candidates(12,
                                                          f'data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'out/{sub_name}.csv.gz')


def graph_submission_walk_len():
    """
    create the submissions for the graph embeddings that use undirected edges
    from customer to item and that vary the walk length and interval duration.
    :return: nothing
    """
    for n_weeks in tqdm([5, 3, 2, 1]):
        for walk_len in [15, 10, 8, 5, 3, 2, 1]:
            sub_name = f'embedding_week_106_nr_{n_weeks}_walk_len_{walk_len}'

            c_candidates = get_graph_embedding_candidates(12,
                                                          f'data/{sub_name}.npy')

            # write the predictions
            write_dict(c_candidates, f'out/{sub_name}.csv.gz')


if __name__ == '__main__':
    # these are the submissions for the graph embeddings that can be uploaded if
    # the week is 106
    # graph_submission_original()

    graph_submission_walk_nr()
    graph_submission_walk_len()

    graph_submission_temporal_naive()
    graph_submission_temporal_improved()
