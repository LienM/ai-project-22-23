import pandas as pd
import pickle
from utils import *

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None


def gc_popularity(articles, customers, transactions, candidates = None, period=-1, k=1000):
    max_day = transactions['t_dat'].max()
    if period == -1:
        min_day = transactions['t_dat'].min()
    else:
        min_day = max_day - pd.Timedelta(days=period)
    transactions = transactions[(transactions['t_dat'] > min_day) & (transactions['t_dat'] <= max_day)]
    # get top k popular items
    top_k = transactions['article_id'].value_counts().rename_axis('article_id').reset_index(name='counts').head(k)

    # make a dict with entry for each customer
    if candidates is None:
        candidates = {}
        for customer_id in customers['customer_id'].values:
            candidates[customer_id] = top_k['article_id'].values
    else:
        for customer_id in candidates.keys():
            candidates[customer_id] = candidates[customer_id].append(top_k['article_id'].values)
    return candidates


def generate_candidates(articles, customers, transactions, method, **kwargs):
    candidates = gc_popularity(articles, customers, transactions, **kwargs)

    return candidates


def get_data_from_canditates(candidates, customer_id, articles, customers, transactions):
    week = transactions['week'].max() + 1
    articles = articles.loc[articles['article_id'].isin(candidates)]
    customers = customers[customers['customer_id'] == customer_id]

    # concatenate articles and customers
    samples = articles.merge(customers, how='cross')
    # add week
    samples['week'] = week
    # drop customer_id
    samples.drop('customer_id', axis=1, inplace=True)

    return samples
