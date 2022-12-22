import os
import warnings

import numpy as np
import pandas as pd
import pdcast as pdc
from tqdm import tqdm

# get from environment variable
DATA_DIR = os.environ.get('DATA_DIR', '../../data/')


def load_data(read_articles=True, read_customers=True, read_transactions=True, frac=1, seed=42, verbose=True):
    """
    Load data from csv files
    :param read_articles: Boolean, whether to read articles.csv
    :param read_customers: Boolean, whether to read customers.csv
    :param read_transactions: Boolean, whether to read transactions.csv
    :param frac: Float, fraction of data to read
    :param seed: Int, seed for random sampling
    :param verbose: Boolean, whether to print progress
    :return: Dictionary with dataframes
    """
    articles, customers, transactions = None, None, None
    if verbose:
        print("\nLoading articles", end='')
    if read_articles:
        if os.path.exists(f'{DATA_DIR}articles.feather'):
            articles = pd.read_feather(f'{DATA_DIR}articles.feather')
        else:
            articles = pd.read_csv(f'{DATA_DIR}articles.csv')
            articles.to_feather(f'{DATA_DIR}articles.feather')

    if verbose:
        print("\rLoading customers", end='')
    if read_customers:
        if os.path.exists(f'{DATA_DIR}customers.feather'):
            customers = pd.read_feather(f'{DATA_DIR}customers.feather')
        else:
            customers = pd.read_csv(f'{DATA_DIR}customers.csv')
            customers.to_feather(f'{DATA_DIR}customers.feather')

    if verbose:
        print("\rLoading transactions", end='')
    if read_transactions:
        if os.path.exists(f'{DATA_DIR}transactions.feather'):
            transactions = pd.read_feather(f'{DATA_DIR}transactions.feather')
        else:
            transactions = pd.read_csv(f'{DATA_DIR}transactions_train.csv')
            transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d')
            transactions.to_feather(f'{DATA_DIR}transactions.feather')

    if frac < 1 and read_transactions and read_customers and read_articles:
        # sample customers
        customers = customers.sample(frac=frac, random_state=seed)
        customers.reset_index(drop=True, inplace=True)
        # sample transactions
        transactions = transactions[transactions['customer_id'].isin(customers['customer_id'])]
        transactions.reset_index(drop=True, inplace=True)
        # sample articles
        articles = articles[articles['article_id'].isin(transactions['article_id'])]
        articles.reset_index(drop=True, inplace=True)
    if verbose:
        print("\r", end='')
    return {'articles': articles, 'customers': customers, 'transactions': transactions}


def test_train_split(samples, val_period=1):
    """
    Split samples into train and validation sets
    :param samples: DataFrame, samples to split
    :param val_period: Int, number of weeks to use for validation
    :return: Tuple of DataFrames
    """
    max_date = samples['t_dat'].max()
    max_train_date = max_date - pd.DateOffset(weeks=val_period)
    train = samples[samples['t_dat'] <= max_train_date]
    val = samples[samples['t_dat'] > max_train_date]

    # transform validation to customer_id, prediction format
    val = val.groupby('customer_id')['article_id'].apply(list).reset_index(name='prediction')
    return train, val


def dict_to_df(d, to_string=True):
    """
    Convert dictionary to DataFrame
    :param d: Dictionary
    :param to_string: Boolean, whether to convert list to string (useful for submission)
    :return:
    """
    if to_string:
        for key in d:
            d[key] = ' '.join([f"0{x}" for x in d[key]])

    df = pd.DataFrame(list(d.items()), columns=['customer_id', 'prediction'])
    return df


def write_submission(results, append=False, verbose=True):
    """
    Write submission file
    :param results: Dictionary or DataFrame with results
    :param append: Boolean, whether to append to existing file
    :param verbose: Boolean, whether to print progress
    :return:
    """
    if verbose:
        print("Writing submission", end='')
    if type(results) == dict:
        results = dict_to_df(results)

    # check if columns are in correct order
    if results.columns[0] != 'customer_id':
        # swap column positions
        columns_titles = ["customer_id", "prediction"]
        results = results.reindex(columns=columns_titles)

    file = f'{DATA_DIR}submission.csv'

    if type(results['prediction'].values[0]) == list:
        results['prediction'] = results['prediction'].apply(lambda x: ' '.join([f"0{y}" for y in x]))

    if append:
        results.to_csv(file, mode='a', index=False, header=False)
    else:
        results.to_csv(file, index=False, header=True)
    if verbose:
        print("\r", end='')


def map_at_12(predictions: pd.DataFrame, ground_truth: pd.DataFrame, verbose=True):
    """
    Compute MAP@12 metric
    :param predictions: DataFrame with predictions
    :param ground_truth: DataFrame with ground truth
    :param verbose: Boolean, whether to print progress
    :return: Float, MAP@12 score
    """
    predictions = predictions.set_index(['customer_id'])
    aps = []
    gt_dict = ground_truth.to_dict('records')
    pbar = tqdm(gt_dict, leave=False) if verbose else gt_dict
    pred_dict = predictions.to_dict('index')
    for row in pbar:
        pred_row = pred_dict[row['customer_id']]
        pred = pred_row['prediction']
        gt = row['prediction']
        if type(gt) == str:
            gt = map(int, gt.split(' '))
        if type(pred) == str:
            pred = map(int, pred.split(' '))
        hits = 0
        ap = 0
        for i, p in enumerate(pred):
            relevance = 1 if p in gt else 0
            hits += relevance
            precision = hits / (i + 1)
            ap += precision * relevance
            if i == 11:
                break
        aps.append(ap / min(len(gt), 12))
        if verbose:
            # pbar.set_description(f"Evaluating using MAP@12: {np.mean(aps)}")
            pbar.set_description(f"Evaluating using MAP@12")
    return np.mean(aps)


def candidates_recall_test(data, ground_truth: pd.DataFrame, count_missing=True, verbose=True):
    """
    Compute recall of candidates
    :param data: DataFrame with candidates
    :param ground_truth: DataFrame with ground truth
    :param count_missing: Boolean, whether to count missing customers
    :param verbose: Boolean, whether to print progress
    :return: Float, recall score
    """
    # get candidates
    samples = data['samples']
    test_week = data['test_week']
    test = samples[samples.week == test_week].drop_duplicates(
        ['customer_id', 'article_id'])
    candidates = test.groupby('customer_id')['article_id'].apply(list).reset_index()

    recalls = []
    gt_dict = ground_truth.to_dict('records')
    pbar = tqdm(gt_dict, leave=False) if verbose else gt_dict
    # group articles by customer
    for row in pbar:
        customer_id = row['customer_id']
        if customer_id not in candidates['customer_id'].values:
            if count_missing: recalls.append(0)
            continue
        gt = row['prediction']
        c_candidates = candidates[candidates['customer_id'] == customer_id]['article_id'].values[0]
        intersection = set(gt).intersection(set(c_candidates))
        recall = len(intersection) / len(gt)
        recalls.append(recall)
        if verbose:
            pbar.set_description(f"Evaluating recall")

    return np.mean(recalls)


def merge_downcast(df1, df2, **kwargs):
    """
    Merge two DataFrames and downcast to the smallest possible type using pandas-downcast
    :param df1: DataFrame
    :param df2: DataFrame
    :param kwargs: Keyword arguments for pd.merge
    :return: Merged DataFrame
    """
    df = pd.merge(df1, df2, **kwargs)
    try:
        dcdf = pdc.downcast(df)
        return dcdf
    except:
        warnings.warn(f"Downcasting failed")
        return df


def concat_downcast(dfs, **kwargs):
    """
    Concatenate DataFrames and downcast to the smallest possible type using pandas-downcast
    :param dfs: List of DataFrames
    :param kwargs: Keyword arguments for pd.concat
    :return: Concatenated DataFrame
    """
    df = pd.concat(dfs, **kwargs)
    try:
        dcdf = pdc.downcast(df)
        return dcdf
    except:
        warnings.warn(f"Downcasting failed")
        return df


def make_purchase_history(transactions):
    """
    Make purchase history from transactions
    :param transactions: DataFrame with transactions
    :return: DataFrame with purchase history
    """
    m = transactions.drop_duplicates(['customer_id', 'article_id']).groupby(['customer_id', 'week'])[
        'article_id'].apply(list).reset_index(name='last_week')
    m = m.sort_values(['customer_id', 'week']).reset_index(drop=True)
    m['week'] += 1
    # append previous weeks
    m['purchase_history'] = m.groupby('customer_id')['last_week'].apply(lambda x: x.cumsum())
    m = m.drop('last_week', axis=1)
    # downcast
    m = pdc.downcast(m)
    return m[['customer_id', 'week', 'purchase_history']]
