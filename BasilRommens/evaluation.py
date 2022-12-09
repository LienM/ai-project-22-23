import time

import numpy as np
import tqdm

import joblib
import pandas as pd


def precision_at_k(top_rel_articles, top_rec_articles, k):
    top_k_predictions = set(top_rec_articles[:k])
    return len(top_k_predictions.intersection(set(top_rel_articles))) / k


def relevance_at_k(top_rel_articles, top_rec_articles, k):
    if len(top_rec_articles):
        return 0
    return top_rec_articles[k] in top_rel_articles


def map_at_k(rel, rec, k):
    # outer sum value
    sum_U = 0
    U_count = 0
    for rec_row in tqdm.tqdm(list(rec.iterrows())):
        # get the uid of the rec_row
        c_id = rec_row[1]['customer_id']

        # get the top k user predictions
        rel_customer = rel[rel['customer_id'] == c_id]
        if not len(rel_customer):
            continue
        rel_articles = np.array(rel_customer['prediction'].values[0].split(' ')[:k])
        rel_articles.astype(int)
        rel_articles = list(rel_articles)

        # inner sum value
        sum_k = 0

        # get the relevant items for the rec_row
        rec_articles = np.array(rec_row[1]['prediction'].split(' '))
        rec_articles.astype(int)
        rec_articles = list(rec_articles)

        # get the # of recommendations and the # of real relevant articles
        n = len(rec_articles)
        m = len(rel_articles)

        # go over all the recommendations
        for at_k in range(1, min(n, k) + 1):
            p = precision_at_k(rel_articles, rec_articles, at_k)
            relev = relevance_at_k(rel_articles, rec_articles, at_k)
            sum_k += p * relev

        if len(rel_articles):
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
    rec = pd.read_csv("../out/age_bin_2.csv.gz")
    transactions = pd.read_feather("../data/transactions.feather")[
        ['customer_id', 'article_id', 'week', 't_dat']]
    val_week = 106
    rel = prepare_transactions(transactions, val_week)
    start = time.time()
    print(map_at_k(rel, rec, 12))
    print(f'done map_at_k in {time.time() - start} seconds')
