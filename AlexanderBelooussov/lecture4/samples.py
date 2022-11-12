import numpy as np
import pandas as pd
from tqdm import tqdm
from recpack_samples import *

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None


def previous_week_article_info(data):
    """
    Data about sales in previous week(s) for each article and week
    :param data:
    :return:
    """
    transactions = data['transactions']

    # cross join of article ids and weeks
    weeks = transactions.week.unique()
    article_ids = transactions.article_id.unique()
    weeks, article_ids = np.meshgrid(weeks, article_ids)
    weeks = weeks.flatten()
    article_ids = article_ids.flatten()
    article_week_info = pd.DataFrame({'week': weeks, 'article_id': article_ids})

    # =====bestseller rank=====
    mean_price = transactions \
        .groupby(['week', 'article_id'])['price'].mean()

    weekly_sales_rank = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='min', ascending=False) \
        .rename('bestseller_rank').astype('int16')

    previous_week_rank = pd.merge(weekly_sales_rank, mean_price, on=['week', 'article_id']).reset_index()
    # previous_week_rank.week += 1
    article_week_info = pd.merge(article_week_info, previous_week_rank, on=['week', 'article_id'], how='left')
    article_week_info.bestseller_rank.fillna(9999, inplace=True)
    article_week_info['bestseller_rank'] = pd.to_numeric(article_week_info['bestseller_rank'], downcast='integer')
    article_week_info.price.fillna(-1, inplace=True)

    # =====sales in last week=====
    weekly_sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .rename('1w_sales').astype('int16').reset_index()
    # weekly_sales.week += 1

    article_week_info = pd.merge(article_week_info, weekly_sales, on=['week', 'article_id'], how='left')
    article_week_info['1w_sales'].fillna(0, inplace=True)
    article_week_info['1w_sales'] = pd.to_numeric(article_week_info['1w_sales'], downcast='integer')

    # =====sales in last 4 weeks=====
    article_week_info['4w_sales'] = article_week_info.groupby('article_id')['1w_sales'].rolling(4,
                                                                                                min_periods=1).sum().reset_index(
        0, drop=True)
    article_week_info['4w_sales'] = pd.to_numeric(article_week_info['4w_sales'], downcast='integer')
    # =====sales in all last weeks=====
    article_week_info['all_sales'] = article_week_info.groupby('article_id')['1w_sales'].cumsum()
    article_week_info['all_sales'] = pd.to_numeric(article_week_info['all_sales'], downcast='integer')
    # =====bestseller rank in last 4 weeks=====
    bestseller_4w = article_week_info.groupby('week')['4w_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_4w').astype('int16').reset_index()
    article_week_info['bestseller_4w'] = bestseller_4w['bestseller_4w']
    # =====bestseller rank in all last weeks=====
    bestseller_all = article_week_info.groupby('week')['all_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_all').astype('int16').reset_index()
    article_week_info['bestseller_all'] = bestseller_all['bestseller_all']

    article_week_info['week'] += 1  # shift week so that it is about the previous week
    # print(article_week_info.head(100))
    return article_week_info


def samples(data, n_train_weeks=12, n=12):
    """
    Generate samples (positive, negative, candidates)
    :param data:
    :param n_train_weeks:
    :param n: Number of samples per method, higher = higher recall
    :return:
    """
    bestseller_types = ['bestseller_rank', 'bestseller_4w', 'bestseller_all']
    # bestseller_types = ['bestseller_rank']

    # ========================================================
    # limit scope to train weeks
    transactions = data['transactions']
    test_week = transactions.week.max() + 1
    if n_train_weeks > 0:
        transactions = transactions[transactions.week > transactions.week.max() - n_train_weeks]
        data['transactions'] = transactions
    # ========================================================
    # gather info about previous weeks
    previous_week_info = previous_week_article_info(data)

    # repurchasing
    c2weeks = transactions.groupby('customer_id')['week'].unique()
    c2weeks2shifted_weeks = {}
    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

    candidates_last_purchase = transactions.copy()

    weeks = []
    for i, (c_id, week) in enumerate(zip(transactions['customer_id'], transactions['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])

    candidates_last_purchase.week = weeks

    # mean_price = transactions \
    #     .groupby(['week', 'article_id'])['price'].mean()
    #
    # sales = transactions \
    #     .groupby('week')['article_id'].value_counts() \
    #     .groupby('week').rank(method='dense', ascending=False) \
    #     .groupby('week').head(12).rename('bestseller_rank').astype('int16')
    #
    # bestsellers_previous_week = pd.merge(sales, mean_price, on=['week', 'article_id']).reset_index()
    # bestsellers_previous_week.week += 1

    # get all customers that made a purchase and the week of their last purchase
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()
    # get all customers that made a purchase
    test_set_transactions = unique_transactions.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_transactions.week = test_week
    bestseller_candidates = pd.DataFrame()
    for bestseller_type in bestseller_types:
        # for customers who made purchases, generate negative samples based on the popular items at the time
        # get the most popular items from the previous week
        bestsellers_previous_week = previous_week_info[previous_week_info[bestseller_type] <= n] \
            [['week', 'article_id', 'bestseller_rank', 'price']].sort_values(['week', 'bestseller_rank'])

        # join popular items and customers + purchase weeks
        candidates_bestsellers = pd.merge(
            unique_transactions,
            bestsellers_previous_week,
            on='week',
        )

        # join popular items and customers
        candidates_bestsellers_test_week = pd.merge(
            test_set_transactions,
            bestsellers_previous_week,
            on='week'
        )
        bestseller_candidates = pd.concat([bestseller_candidates,
                                           candidates_bestsellers,
                                           candidates_bestsellers_test_week])
    # drop rank, to be added later
    bestseller_candidates.drop(columns='bestseller_rank', inplace=True)

    # some initial recpack candidates
    # TODO also do for negative samples
    recpack = recpack_samples(transactions, n)
    recpack['week'] = test_week
    recpack_candidates = pd.merge(
        test_set_transactions,
        recpack,
        on=['week', 'customer_id']
    )
    # add price
    recpack_candidates = recpack_candidates.merge(previous_week_info[['week', 'article_id', 'price']], on=['week', 'article_id'], how='left')

    # ===================================================================================================
    # set purchased for positive samples
    transactions['purchased'] = 1
    # combine transactions and candidates
    samples = pd.concat([transactions,
                         candidates_last_purchase,
                         bestseller_candidates,
                         recpack_candidates
                         ])
    # set purchased to 0 for candidates and negative samples
    samples.purchased.fillna(0, inplace=True)
    # only keep the first occurrence of a customer-week-article combination
    # this will keep bought articles since they are concatenated first
    samples.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)

    # print some statistics about our samples
    p_sum = samples[samples['week'] != test_week].purchased.sum()
    p_mean = samples[samples['week'] != test_week].purchased.mean()
    p_neg = samples[samples['week'] != test_week].purchased.value_counts()[0]
    n_candidates = samples[samples['week'] == test_week].shape[0]
    print(f"RATIO OF POSITIVE SAMPLES: {p_mean * 100:.2f}%, {1}:{p_neg / p_sum:.2f}")
    print(f"#CANDIDATES: {n_candidates}")

    # add info about sales in previous weeks
    samples = pd.merge(
        samples,
        previous_week_info[['week', 'article_id',
                            'bestseller_rank', 'bestseller_4w', 'bestseller_all',
                            '1w_sales', '4w_sales', 'all_sales']],
        on=['week', 'article_id'],
        how='left'
    )
    # finalise samples
    samples = samples[samples.week != samples.week.min()]  # remove first week due to lack of information
    samples = pd.merge(samples, data['articles'], on='article_id', how='left')  # merge article info
    samples = pd.merge(samples, data['customers'], on='customer_id', how='left')  # merge customer info

    # fix dtypes
    samples['week'] = pd.to_numeric(samples['week'], downcast='integer')
    samples['bestseller_rank'] = pd.to_numeric(samples['bestseller_rank'], downcast='integer')
    samples['bestseller_4w'] = pd.to_numeric(samples['bestseller_4w'], downcast='integer')
    samples['bestseller_all'] = pd.to_numeric(samples['bestseller_all'], downcast='integer')
    samples['1w_sales'] = pd.to_numeric(samples['1w_sales'], downcast='integer')
    samples['4w_sales'] = pd.to_numeric(samples['4w_sales'], downcast='integer')
    samples['all_sales'] = pd.to_numeric(samples['all_sales'], downcast='integer')
    samples['purchased'] = pd.to_numeric(samples['purchased'], downcast='integer')
    samples['price'] = pd.to_numeric(samples['price'], downcast='float')

    samples.sort_values(['week', 'customer_id'], inplace=True)
    samples.reset_index(drop=True, inplace=True)
    data['samples'] = samples
    # print(samples.head(200).sort_values(by=['article_id', 'week'], inplace=False))
    # print(f"Samples shape: {samples.shape}")
    # print(samples[samples.week == test_week].head(200).sort_values(by=['article_id', 'week'], inplace=False))
    # raise Exception('stop')
    data['test_week'] = test_week
    data['article_week_info'] = previous_week_info
    return data
