import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = 'data/'


def load_data(read_articles=True, read_customers=True, read_transactions=True, frac=1):
    print("\nLoading articles", end='')
    articles = pd.read_csv(f'{DATA_DIR}articles.csv') if read_articles else None
    print("\rLoading customers", end='')
    customers = pd.read_csv(f'{DATA_DIR}customers.csv') if read_customers else None
    print("\rLoading transactions", end='')
    transactions = pd.read_csv(f'{DATA_DIR}transactions_train.csv') if read_transactions else None
    if read_transactions:
        transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d')
        transactions = transactions.sort_values(by=['t_dat'])
    if frac < 1 and read_transactions and read_customers and read_articles:
        # sample customers
        customers = customers.sample(frac=frac)
        customers.reset_index(drop=True, inplace=True)
        # sample transactions
        transactions = transactions[transactions['customer_id'].isin(customers['customer_id'])]
        transactions.reset_index(drop=True, inplace=True)
        # sample articles
        articles = articles[articles['article_id'].isin(transactions['article_id'])]
        articles.reset_index(drop=True, inplace=True)
    print("\r", end='')
    return articles, customers, transactions


def test_train_split(samples, val_period=1):
    max_week = samples['week'].max()
    min_val_week = max_week - val_period + 1
    train = samples[samples['week'] < min_val_week]
    val = samples[samples['week'] >= min_val_week]

    # transform validation to customer_id, prediction format
    val = val.groupby('customer_id')['article_id'].apply(list).reset_index(name='prediction')
    return train, val


def dict_to_df(d, to_string=True):
    if to_string:
        for key in d:
            d[key] = ' '.join([f"0{x}" for x in d[key]])

    df = pd.DataFrame(list(d.items()), columns=['customer_id', 'prediction'])
    return df


def write_submission(results, append=False):
    print("Writing submission", end='')
    if type(results) == dict:
        results = dict_to_df(results)

    # check if columns are in correct order
    if results.columns[0] != 'customer_id':
        # swap column positions
        columns_titles = ["customer_id", "prediction"]
        results = results.reindex(columns=columns_titles)

    file = f'{DATA_DIR}submission.csv'
    if append:
        results.to_csv(file, mode='a', index=False, header=False)
    else:
        results.to_csv(file, index=False, header=True)
    print("\r", end='')


def map_at_12(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    predictions = predictions.set_index(['customer_id'])
    aps = []
    gt_dict = ground_truth.to_dict('records')
    pbar = tqdm(gt_dict, leave=False)
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
        pbar.set_description(f"Evaluating using MAP@12: {np.mean(aps)}")
    return np.mean(aps)
