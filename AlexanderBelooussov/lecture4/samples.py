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
    pos = pos[pos['t_dat'] > max_date - pd.Timedelta(weeks=period)][
        ['customer_id', 'article_id', 'week']]
    del transactions

    # add label
    pos['y'] = 1

    return pos


def neg_samples(articles, customers, transactions, purchase_hist, verbose=True, period=999):
    max_date = transactions['t_dat'].max()
    min_date = max_date - pd.Timedelta(weeks=period)
    # generate random articles and weeks for each customer

    # repeat average number of purchases per customer
    n = int(purchase_hist.shape[0] / customers.shape[0])
    neg = pd.DataFrame()
    transactions = transactions[transactions['t_dat'] > min_date]
    repeats = tqdm(range(n), desc="Generating negative samples", leave=False) if verbose else range(n)
    for i in repeats:
        tmp = customers[['customer_id']].copy()
        # sample
        tmp['article_id'] = np.random.choice(transactions['article_id'], size=tmp.shape[0])
        tmp['week'] = np.random.choice(transactions['week'], size=tmp.shape[0])
        # remove already purchased articles
        tmp = tmp[~tmp[['customer_id', 'article_id']].isin(purchase_hist[['customer_id', 'article_id']]).all(axis=1)]
        neg = neg.append(tmp)

    neg['y'] = 0

    # print(neg.head(50))
    return neg


def generate_samples(articles, customers, transactions, force, write=True, verbose=True, **kwargs):
    if verbose:
        print(f"Generating samples...", end="")
    if not os.path.exists('pickles/samples.pkl') or force:
        # resample if pickle file is not found
        samples = pos_samples(articles, customers, transactions, **kwargs)
        neg = neg_samples(articles, customers, transactions, samples, verbose, **kwargs)
        # append positive and negative samples
        samples = samples.append(neg)
        # shuffle
        samples = samples.sample(frac=1).reset_index(drop=True)
        # join with articles and customers
        samples = samples.merge(customers, on='customer_id', how='left')
        samples = samples.merge(articles, on='article_id', how='left')
        if write:
            samples.to_pickle('pickles/samples.pkl')
    else:
        samples = pd.read_pickle('pickles/samples.pkl')

    # print(samples.head(50))
    if verbose:
        print("\r", end="")
    return samples
