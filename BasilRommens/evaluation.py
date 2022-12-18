import time
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

import joblib
import pandas as pd


def precision_at_k(top_rel_articles, top_rec_articles, k):
    count = sum([1 for article in top_rec_articles[:k + 1] if article in top_rel_articles])
    return count / (k + 1)


def relevance_at_k(top_rel_articles, top_rec_articles, k):
    if not len(top_rec_articles):
        return 0
    return top_rec_articles[k] in top_rel_articles


def map_at_k(rel, rec, k):
    # outer sum value
    sum_U = 0
    U_count = 0

    # setting index of rel and rec to customer_id
    rel = rel.set_index('customer_id')
    rec = rec.set_index('customer_id')
    # join both rel and rec
    joined = rel.join(rec, how='left', lsuffix='_rel', rsuffix='_rec')

    # iteraterate over all customers to determine map at k scores
    for rec_row in list(joined.iterrows()):
        if rec_row[1]['prediction_rec'] is None:
            U_count += 1
            continue

        rel_articles = np.array(rec_row[1]['prediction_rel'].split(' '))
        rel_articles = rel_articles.astype(int)
        rel_articles = list(rel_articles)

        # inner sum value
        sum_k = 0

        # get the relevant items for the rec_row
        rec_articles = np.array(rec_row[1]['prediction_rec'].split(' '))
        rec_articles = rec_articles.astype(int)
        rec_articles = list(rec_articles)

        # get the # of recommendations and the # of real relevant articles
        n = len(rec_articles)
        m = len(rel_articles)

        # go over all the recommendations
        for at_k in range(0, min(n, k)):
            p = precision_at_k(rel_articles, rec_articles, at_k)
            relev = relevance_at_k(rel_articles, rec_articles, at_k)
            sum_k += p * relev

        sum_U += sum_k / min(m, 12)

        U_count += 1

    return sum_U / U_count


def prepare_transactions(transactions, val_week):
    transactions = transactions[transactions['week'] == val_week]

    transactions = transactions.sort_values(by=['t_dat'])
    customer_ids = transactions['customer_id'].unique()

    c_id_n_a_ids = {'customer_id': list(), 'prediction': list()}
    for customer_id in customer_ids:
        article_ids = transactions[transactions['customer_id'] == customer_id][
                          'article_id'].values[:12]  # 12 is cut-off
        article_ids_string = ' '.join(map(lambda x: '0' + str(x), article_ids))
        c_id_n_a_ids['prediction'].append(article_ids_string)
        c_id_n_a_ids['customer_id'].append(customer_id)

    customer_encoder = joblib.load('../data/customer_encoder.joblib')
    c_id_n_a_ids['customer_id'] = customer_encoder.inverse_transform(
        c_id_n_a_ids['customer_id'])

    return pd.DataFrame(c_id_n_a_ids)


if __name__ == "__main__":
    transactions = pd.read_feather('../data/transactions.feather')[
        ['customer_id', 'article_id', 'week', 't_dat']]
    map_at_k_dict = dict()
    for last_week in [105, 104, 103, 102]:
        rel = prepare_transactions(transactions, last_week + 1)
        map_at_k_dict[last_week] = dict()
        for n_weeks in [40, 20, 10, 5, 3, 2, 1]:
            sub_name = f'graph_embeddings_week_{last_week}_nr_{n_weeks}.csv.gz'
            rec = pd.read_csv(f'../out/{sub_name}.csv.gz')
            start = time.time()
            map_at_k_dict[last_week][n_weeks] = map_at_k(rel, rec, 12)
            print(f'done map_at_k in {time.time() - start} seconds')
    print(map_at_k_dict)
