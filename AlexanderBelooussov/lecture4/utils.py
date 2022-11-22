import os

import numpy as np
import pandas as pd
import pdcast as pdc
from tqdm import tqdm

# get from environment variable
DATA_DIR = os.environ.get('DATA_DIR', '../../data/')


def get_max_date(data):
    max_row = data.sort_values(by=['year', 'month', 'day'], inplace=True, ascending=False).head(1)[['year', 'month', 'day']]
    year = max_row['year'].values[0]
    month = max_row['month'].values[0]
    day = max_row['day'].values[0]
    return {'year': year,
            'month': month,
            'day': day,
            'date': pd.to_datetime(f"{year}-{month}-{day}", format='%Y-%m-%d')}


def load_data(read_articles=True, read_customers=True, read_transactions=True, frac=1, seed=42, verbose=True):
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
    max_date = samples['t_dat'].max()
    max_train_date = max_date - pd.DateOffset(weeks=val_period)
    train = samples[samples['t_dat'] <= max_train_date]
    val = samples[samples['t_dat'] > max_train_date]

    # transform validation to customer_id, prediction format
    val = val.groupby('customer_id')['article_id'].apply(list).reset_index(name='prediction')
    return train, val


def dict_to_df(d, to_string=True):
    if to_string:
        for key in d:
            d[key] = ' '.join([f"0{x}" for x in d[key]])

    df = pd.DataFrame(list(d.items()), columns=['customer_id', 'prediction'])
    return df


def write_submission(results, append=False, verbose=True):
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
    df = pd.merge(df1, df2, **kwargs)
    try:
        dcdf = pdc.downcast(df)
        return dcdf
    except:
        print(f"Downcasting failed")
        return df

def concat_downcast(dfs, **kwargs):
    # for df in dfs[1:]:
    #     assert df.shape[0] > 0, "Empty dataframe"
    df = pd.concat(dfs, **kwargs)
    try:
        dcdf = pdc.downcast(df)
        return dcdf
    except:
        print(f"Downcasting failed")
        return df


def make_purchase_history(transactions):
    m = transactions.drop_duplicates(['customer_id', 'article_id']).groupby(['customer_id', 'week'])['article_id'].apply(list).reset_index(name='last_week')
    m = m.sort_values(['customer_id', 'week']).reset_index(drop=True)
    m['week'] += 1
    # append previous weeks
    m['purchase_history'] = m.groupby('customer_id')['last_week'].apply(lambda x: x.cumsum())
    m = m.drop('last_week', axis=1)
    # downcast
    m = pdc.downcast(m)
    return m[['customer_id', 'week', 'purchase_history']]


def add_similarity(samples, purchase_hist, sim, sim_index):
    """
    Add similarity to samples
    Pretty slow but oh well
    :param samples:
    :param purchase_hist:
    :param sim:
    :param sim_index:
    :return:
    """
    print(f"Adding similarity to samples...")
    purchase_hist = purchase_hist.set_index(['customer_id', 'week'])
    # turn items of purchase history into indices
    purchase_hist['purchase_history'] = purchase_hist['purchase_history'].apply(lambda x: [sim_index[item] for item in x])
    hist = purchase_hist.to_dict("index")
    min_week = samples.week.min()

    def get_sim(row):
        article_idx = sim_index[row[1]]
        key = (row[0], row[2])
        while key not in hist:
            if key[1] < min_week:
                return float(0)
            key = (key[0], key[1] - 1)
        return sim[article_idx, hist[key]['purchase_history']].mean()

    # try:
    #     s = samples[['customer_id', 'article_id', 'week']]. \
    #         swifter.progress_bar(enable=True, desc=f"Calculating similarity"). \
    #         apply(lambda x: get_sim(x), axis=1, raw=True)
    # except Exception as e:
    #     print(f"Swifter failed")
    #     print(e)
    tqdm.pandas()
    s = samples[['customer_id', 'article_id', 'week']].progress_apply(lambda x: get_sim(x), axis=1,
                                                                                   raw=True)

    return s


