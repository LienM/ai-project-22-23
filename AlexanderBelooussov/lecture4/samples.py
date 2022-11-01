import os

import pandas as pd
import swifter
import numpy as np
from tqdm import tqdm


def merge_frames(df1, df2, on):
    # https://stackoverflow.com/questions/47386405/memoryerror-when-i-merge-two-pandas-data-frames
    df_result = pd.DataFrame(columns=(df1.columns.append(df2.columns)).unique())
    rows = df2.shape[0]
    df_result.to_csv("df3.csv", index_label=False)
    df2.to_csv("df2.csv", index=False)
    # deleting df2 to save memory
    del (df2)

    def preprocess(x):
        df2 = pd.merge(df1, x, on=on, how='right')
        df2.to_csv("df3.csv", mode="a", header=False, index=False)

    reader = pd.read_csv("df2.csv", chunksize=rows // 10)

    [preprocess(r) for r in tqdm(reader, desc="Merging dataframes", leave=False)]


def pos_samples(articles, customers, transactions, period=999):
    # dict of purchased articles for each customer
    max_date = transactions['t_dat'].max()
    pos = transactions[transactions['customer_id'].isin(customers['customer_id'])]
    pos = pos[pos['t_dat'] > max_date - pd.Timedelta(days=period)][
        ['customer_id', 'article_id', 'week']]
    del transactions

    # add label
    pos['y'] = 1

    return pos


def neg_samples(articles, customers, transactions, purchase_hist, period=999):
    # generate random articles and weeks for each customer

    # repeat average number of purchases per customer
    n = int(purchase_hist.shape[0] / customers.shape[0])
    neg = pd.DataFrame()
    transactions = transactions[transactions['t_dat'] > transactions['t_dat'].max() - pd.Timedelta(days=period)]
    for i in tqdm(range(n), desc="Generating negative samples", leave=False):
        tmp = customers[['customer_id']].copy()
        tmp['article_id'] = tmp['customer_id'].swifter.progress_bar(enable=False).apply(
            lambda x: np.random.choice(articles['article_id'].values, 1)[0])
        tmp['week'] = tmp['customer_id'].swifter.progress_bar(enable=False).apply(
            lambda x: np.random.choice(transactions['week'].values, 1)[0])
        # remove already purchased articles
        tmp = tmp[~tmp[['customer_id', 'article_id']].isin(purchase_hist[['customer_id', 'article_id']]).all(axis=1)]
        neg = neg.append(tmp)

    # join with articles and customers
    # neg = neg.merge(customers, on='customer_id', how='left')
    # neg = neg.merge(articles, on='article_id', how='left')

    neg['y'] = 0

    # print(neg.head(50))
    return neg


def generate_samples(articles, customers, transactions, force, **kwargs):
    print(f"Generating samples...", end="")
    if not os.path.exists('pickles/samples.pkl') or force:
        # resample if pickle file is not found
        samples = pos_samples(articles, customers, transactions, **kwargs)
        neg = neg_samples(articles, customers, transactions, samples, **kwargs)
        # append positive and negative samples
        samples = samples.append(neg)
        # shuffle
        samples = samples.sample(frac=1).reset_index(drop=True)
        # join with articles and customers

        # print(samples.head(50))

        # print samples, customer, articles memory usage
        # print("Samples memory usage: {:.2f} MB".format(samples.memory_usage().sum() / 1024 ** 2))
        # print("Customers memory usage: {:.2f} MB".format(customers.memory_usage().sum() / 1024 ** 2))
        # print("Articles memory usage: {:.2f} MB".format(articles.memory_usage().sum() / 1024 ** 2))
        #
        # # print shapes
        # print("Samples shape: {}".format(samples.shape))
        # print("Customers shape: {}".format(customers.shape))
        # print("Articles shape: {}".format(articles.shape))

        samples = samples.merge(customers, on='customer_id', how='left')
        samples = samples.merge(articles, on='article_id', how='left')
        samples.to_pickle('pickles/samples.pkl')
    else:
        samples = pd.read_pickle('pickles/samples.pkl')

    # print(samples.head(50))
    print("\r", end="")
    return samples
