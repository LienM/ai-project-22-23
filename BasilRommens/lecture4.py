import collections
import itertools
import time

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
import pandas as pd
import random

from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing

from BasilRommens.cleaning import clean_articles, clean_customers, \
    clean_transactions
from BasilRommens.dataset import prepare_feather_datasets, part_data_set, \
    read_data_set
from BasilRommens.evaluation import map_at_k
from tqdm import tqdm
from lightgbm import LGBMRanker
import lightgbm as lgb
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

from BasilRommens.globals import usable_cols


def popular_predictor(customer_id, top_counts):
    # top_counts = collections.Counter(transactions.groupby('article_id')[
    #                                      'article_id'].count().to_dict()).most_common(12)
    predictions = list(map(lambda x: '0' + str(x[0]), top_counts))
    return ' '.join(predictions)


def get_n_popular_articles(transactions, n):
    grouped_articles = transactions.groupby('article_id')['article_id']
    counted_articles = grouped_articles.count()
    sorted_articles = counted_articles.sort_values(ascending=False)
    top_n_articles = sorted_articles[:n].index
    return list(top_n_articles)


def get_neg_transactions(transactions):
    # add negative samples
    positive_pairs = list(map(tuple, transactions[
        ['customer_id', 'article_id']].drop_duplicates().values))

    # Extract real values
    real_dates = transactions["t_dat"].unique()
    real_customers = transactions["customer_id"].unique()
    # fetch n most popular articles
    real_articles = get_n_popular_articles(transactions, 100)
    real_channels = transactions["sales_channel_id"].unique()
    real_weeks = transactions["week"].unique()
    article_and_price = transactions[["article_id", "price"]].drop_duplicates(
        "article_id").set_index("article_id").squeeze()

    # How many negatives do we need to sample?
    num_neg_pos = transactions.shape[0]

    # Sampling negatives by selecting random users, articles, dates and sales channel:
    # Note: This is quite naive. Some articles may not even have been available at the date we are sampling.
    random.seed(42)

    # Afterwards, we need to remove potential duplicates, so we'll sample too many.
    num_neg_samples = int(num_neg_pos * 1.1)

    # Sample each of the independent attributes.
    neg_dates = np.random.choice(real_dates, size=num_neg_samples)
    neg_articles = np.random.choice(real_articles, size=num_neg_samples)
    neg_customers = np.random.choice(real_customers, size=num_neg_samples)
    neg_channels = np.random.choice(real_channels, size=num_neg_samples)
    neg_weeks = np.random.choice(real_weeks, size=num_neg_samples)
    ordered = np.array([0] * num_neg_samples)
    # Assign to every article a real price.
    neg_prices = article_and_price[neg_articles].values

    neg_transactions = pd.DataFrame(
        [neg_dates, neg_customers, neg_articles, neg_prices, neg_channels,
         neg_weeks, ordered], index=transactions.columns).T

    # Remove random negative samples that actually coincide with positives
    df = neg_transactions[
        ~neg_transactions.set_index(["customer_id", "article_id"]).index.isin(
            positive_pairs)
    ]
    # Remove any excess
    chosen_neg_transactions = df.sample(num_neg_pos)
    return chosen_neg_transactions


def predict_topn(X_reduced, model):
    topn = dict()
    for uid in X_reduced['customer_id'].unique():
        topn[uid] = list()
        for iid in X_reduced['article_id'].unique():
            est = model.predict(uid, iid).est
            if est < 0.95:
                continue
            topn[uid].append((iid, est))

        temp_topn = dict()
        for uid, est in topn.items():
            sorted_est = sorted(est, key=lambda x: x[1])
            temp_topn[uid] = sorted_est[:12]
        topn = temp_topn
        del temp_topn
    return topn


def get_group_sizes(dataset):
    group_sizes = dataset.groupby(['week', 'customer_id'])[
        'article_id'].count().values
    return group_sizes


# copied from
# https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
def get_candidate_bestsellers(transactions, test_week):
    # get the average price of each article in a given week
    mean_price = transactions \
        .groupby(['week', 'article_id'])['price'].mean()

    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='dense', ascending=False) \
        .groupby('week').head(12).rename('bestseller_rank').astype('int8')

    # merge the mean price and the sales
    bestsellers_previous_week = pd.merge(sales, mean_price,
                                         on=['week', 'article_id']) \
        .reset_index()
    bestsellers_previous_week['week'] += 1  # move the week 1 week forward

    # get a single transaction per customer per week so that we can merge
    # bestseller candidates with the transactions of customers who made a
    # purchase in a week
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()

    # merge all the candidates with the transactions of customers who made a
    # purchase
    candidates_bestsellers = pd.merge(
        unique_transactions,
        bestsellers_previous_week,
        on='week',
    )

    # remove duplicate customer ids in unique customerid transactions
    test_set_transactions = unique_transactions.drop_duplicates(
        'customer_id').reset_index(drop=True)
    # set the test week for all the customers
    test_set_transactions.week = test_week

    # merge the candidates for the test week
    candidates_bestsellers_test_week = pd.merge(
        test_set_transactions,
        bestsellers_previous_week,
        on='week'
    )

    # add the test week candidates to the rest of the candidates
    candidates_bestsellers = pd.concat(
        [candidates_bestsellers, candidates_bestsellers_test_week])
    # remove the bestseller rank and if the ordered column
    candidates_bestsellers.drop(columns=['bestseller_rank', 'ordered'],
                                inplace=True)
    return candidates_bestsellers, bestsellers_previous_week


# copied from
# https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
def last_purchase_candidates(transactions):
    c2weeks = transactions.groupby('customer_id')['week'].unique()

    c2weeks2shifted_weeks = {}

    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

    candidates_last_purchase = transactions.copy()

    weeks = []
    for i, (c_id, week) in enumerate(
            zip(transactions['customer_id'], transactions['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])

    candidates_last_purchase['week'] = weeks
    # sales = candidates_last_purchase \
    #     .groupby(['week', 'customer_id'])['t_dat'].rank(method='dense',
    #                                                     ascending=False)

    return candidates_last_purchase


def get_age_interval_idx(age, *intervals):
    if type(intervals[0][0]) != list:
        intervals = [intervals]
    for interval_idx, interval in enumerate(intervals[0]):
        if interval[0] <= age <= interval[1]:
            return interval_idx

    return -1  # default value


def get_interval_ranking(transactions, cur_interval):
    # take only transactions in cur_interval
    interval_transactions = transactions[transactions['age'].isin(cur_interval)]
    ranking = interval_transactions \
        .groupby('product_type_no')['product_type_no'].sum() \
        .rank(method='dense').rename('rank') \
        .astype(np.uint8) \
        .reset_index() \
        .sort_values('rank').head(12)
    return ranking.to_numpy()


def compare_rankings(ranking_1, ranking_2):
    # first item in the
    sim = 0
    for key_2, rank_idx_2 in ranking_2:
        added = False
        for key_1, rank_idx_1 in ranking_1:
            if key_1 == key_2:
                sim += abs(rank_idx_1 - rank_idx_2)
                added = True
                break
        if not added:
            sim += max(len(ranking_1), len(ranking_2)) * 2
    return sim


def smart_merge_intervals(transactions, threshold=100):
    count = transactions \
        .groupby('age')['product_type_no'].value_counts() \
        .rename('count') \
        .reset_index().merge(transactions, on=['age', 'product_type_no'])

    #     .groupby('age').head(10) \
    #     .astype(np.uint8) \

    transactions = transactions.sort_values('age')
    intervals = list()
    cur_interval = list()
    threshold = 100
    sims = list()
    for age in transactions['age'].unique():
        if age == -1:  # skip if default value
            continue
        if not cur_interval:
            cur_interval.append(age)
            continue
        # append age to current interval if age is compatible with cur_interval
        # set
        interval_ranking = get_interval_ranking(count, cur_interval)
        age_ranking = get_interval_ranking(transactions, [age])
        sim = compare_rankings(interval_ranking, age_ranking)
        sims.append(sim)
        if sim < threshold:
            cur_interval.append(age)
        else:
            intervals.append(cur_interval.copy())
            cur_interval.clear()
            cur_interval.append(age)
    # sims = sorted(sims)
    # sns.histplot(x=sims)
    # plt.show()

    correct_intervals = list()
    for interval in intervals:
        new_interval = [interval[0], interval[-1]]
        correct_intervals.append(new_interval)
    return correct_intervals


def age_bin_candidates(transactions, test_week, bin_size=1, intervals=None,
                       threshold=100):
    # add bin size to the age column
    if intervals is None:  # use bin size if no intervals are given
        transactions['age_bin'] = transactions['age'] // bin_size
    elif intervals == 'smart':  # use smart interval algorithm
        transactions['age_bin'] = transactions['age']
        new_transactions = pd.DataFrame()
        for week in transactions['week'].unique():
            weekly_transactions = transactions[transactions['week'] == week]
            intervals = smart_merge_intervals(weekly_transactions, threshold)
            weekly_transactions['age_bin'] = \
                weekly_transactions['age_bin'] \
                    .apply(get_age_interval_idx, args=(intervals,))
            new_transactions = pd.concat(
                [new_transactions, weekly_transactions])
        transactions = new_transactions
        del new_transactions
    else:  # use intervals if they are given
        transactions['age_bin'] = transactions['age']
        transactions['age_bin'] = transactions[['age_bin', 'week']] \
            .apply(get_age_interval_idx, args=(intervals,))
    transactions['age_bin'].astype(np.uint8)

    # TRAINING WEEK CANINATES
    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    bestsellers_previous_week = transactions \
        .groupby(['week', 'age_bin'])['product_type_no'].value_counts() \
        .groupby(['week', 'age_bin']).rank(method='dense', ascending=False) \
        .groupby(['week', 'age_bin']).head(12).rename('age_bestseller_rank') \
        .astype('uint8') \
        .reset_index()
    bestsellers_previous_week.week += 1  # move the week 1 week forward
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['product_type_no']) \
        .copy()
    candidates_bestsellers = pd.merge(
        unique_transactions,
        bestsellers_previous_week,
        on=['week', 'age_bin'],
    )
    candidates_bestsellers.drop(columns=['age_bestseller_rank', 'ordered'],
                                inplace=True)

    # TEST WEEK CANINATES
    # remove duplicate customer ids in unique customerid transactions
    test_set_transactions = unique_transactions.drop_duplicates(
        'customer_id').reset_index(drop=True)
    # set the test week for all the customers
    test_set_transactions.week = test_week

    # merge the candidates for the test week
    candidates_bestsellers_test_week = pd.merge(
        test_set_transactions,
        bestsellers_previous_week,
        on=['week', 'age_bin']
    )
    # add the test week candidates to the rest of the candidates
    candidates_bestsellers = pd.concat(
        [candidates_bestsellers, candidates_bestsellers_test_week])
    # remove the bestseller rank and if the ordered column
    candidates_bestsellers.drop(columns=['age_bestseller_rank', 'ordered'],
                                inplace=True)

    return candidates_bestsellers, bestsellers_previous_week


def get_last_weeks_popularity_el(transactions, group_el, pop_type):
    transactions_up_shift_week = transactions.copy()
    transactions_up_shift_week['week'] += 1
    pop = \
        transactions_up_shift_week.groupby(['week', group_el],
                                           as_index=False)[['ordered']].sum()
    pop[pop_type] = pop['ordered']
    pop = pop.drop('ordered', axis=1)
    transactions = transactions.merge(pop, on=['week', group_el])
    return transactions


def get_last_weeks_popularity(pop_type, transactions):
    if pop_type == 'item_popularity':
        transactions = get_last_weeks_popularity_el(transactions, 'article_id',
                                                    pop_type)
    elif pop_type == 'colour_popularity':
        transactions = get_last_weeks_popularity_el(transactions,
                                                    'colour_group_code',
                                                    pop_type)
    elif pop_type == 'product_type_popularity':
        transactions = get_last_weeks_popularity_el(transactions,
                                                    'product_type_no', pop_type)
    return transactions[pop_type]


def show_feature_importances(model):
    for i in model.feature_importances_.argsort()[::-1]:
        print(usable_cols[i], model.feature_importances_[
            i] / model.feature_importances_.sum())


def get_relevant_cols(articles, customers, transactions):
    # taking the relevant columns of all dataframes
    relevant_article_cols = ['article_id', 'product_type_no',
                             'graphical_appearance_no', 'colour_group_code',
                             'perceived_colour_value_id',
                             'perceived_colour_master_id', 'department_no',
                             'index_group_no', 'section_no', 'garment_group_no']
    relevant_customer_cols = ['customer_id', 'FN', 'Active',
                              'fashion_news_frequency', 'age', 'postal_code']
    relevant_transaction_cols = ['customer_id', 'article_id', 'price',
                                 'sales_channel_id', 'week']
    articles = articles[relevant_article_cols]
    customers = customers[relevant_customer_cols]
    transactions = transactions[relevant_transaction_cols]
    return articles, customers, transactions


def construct_X_y(df, usable_cols):
    X = df[usable_cols]
    try:
        y = df[['ordered']]
    except Exception as e:
        y = None
    return X, y


def last_n_week_transactions(transactions, last_week, n):
    return transactions[transactions.week > last_week - n]


def merge_transactions(transactions, articles, customers):
    transactions = transactions.merge(articles, on='article_id')
    transactions = transactions.merge(customers, on='customer_id')
    transactions = transactions.reset_index(drop=True)
    return transactions


def get_articles_of_product_type(transactions, product_type_no):
    try:
        products = transactions.loc[product_type_no]
    except KeyError:
        return []
    return products['article_id'].tolist()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def change_age(transactions):
    # will 'guess' the age of a customer based on their cosine similarity to
    # an age group
    customers = transactions[transactions['age'] == -1]['customer_id'].unique()
    customer_age_map = {customer: -1 for customer in customers}

    # create age vectors
    all_ages = transactions[transactions['age'] != -1]
    all_age_dict = all_ages \
        .groupby(['age', 'product_type_no'])['product_type_no'] \
        .count() \
        .to_dict()
    product_type_dim = max(transactions['product_type_no'].values) + 1
    age_vectors = {age: np.zeros(product_type_dim)
                   for age in all_ages['age'].unique()}

    for (age, product_type_no), count in tqdm(all_age_dict.items()):
        age_vectors[age][product_type_no] = count

    # create customer vectors
    customer_counts = transactions[transactions['age'] == -1] \
        .groupby(['customer_id', 'product_type_no'])['product_type_no'] \
        .count() \
        .to_dict()

    customer_vectors = {customer: np.zeros(product_type_dim)
                        for customer in customers}
    for (customer_id, product_type_no), count in tqdm(customer_counts.items()):
        customer_vectors[customer_id][product_type_no] = count

    # get age from most similar age group for each customer
    for customer in tqdm(customers):
        customer_vector = customer_vectors[customer]
        similarity = {age: cosine_similarity(customer_vector, age_vector)
                      for age, age_vector in age_vectors.items()}
        customer_age_map[customer] = max(similarity, key=similarity.get)

    # change all transactions based on the map
    neg_ages = transactions[transactions['age'] == -1]
    neg_ages['age'] = neg_ages['customer_id'].map(customer_age_map)
    pos_ages = transactions[transactions['age'] != -1]
    transactions = pd.concat([neg_ages, pos_ages])
    return transactions


def get_bestseller_dict(age_bestsellers_previous_week, transactions, last_week):
    transactions = transactions.set_index('product_type_no')
    bestsellers_last_week = \
        age_bestsellers_previous_week[
            age_bestsellers_previous_week.week == last_week][
            ['product_type_no', 'age_bin']]
    unique_age_bins = bestsellers_last_week.age_bin.unique()
    bestsellers_age_bin_dict = {age_bin: list(bestsellers_last_week[
                                                  bestsellers_last_week[
                                                      'age_bin'] == age_bin][
                                                  'product_type_no'].values) for
                                age_bin in unique_age_bins}

    for age_bin in tqdm(unique_age_bins):
        age_bin_bestsellers = bestsellers_age_bin_dict[age_bin]

        age_bin_bestsellers = \
            [get_articles_of_product_type(transactions, product_type_no)
             for product_type_no in age_bin_bestsellers]
        new_age_bin_bestsellers = []
        for age_bin_bestseller in age_bin_bestsellers:
            if type(age_bin_bestseller) == list:
                new_age_bin_bestsellers.append(age_bin_bestseller)
            else:
                new_age_bin_bestsellers.append([age_bin_bestseller])

        bestsellers_age_bin_dict[age_bin] = \
            list(itertools.chain(*new_age_bin_bestsellers))[:12]

    return bestsellers_age_bin_dict


def get_cid_2_preds(test):
    return test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()


def get_cid_2_age_bin(transactions):
    return transactions.groupby('customer_id')['age_bin'] \
        .first() \
        .to_dict()


def make_and_write_predictions_age(model, test, X_test,
                                   age_bestsellers_previous_week, customers,
                                   customer_encoder, transactions, last_week,
                                   sub_name):
    # get the predictions and convert to dict
    test['preds'] = model.predict(X_test)
    c_id2predicted_article_ids = get_cid_2_preds(test)

    # get the bestsellers for each age bin and convert to a dict
    bestsellers_age_bin_dict = get_bestseller_dict(
        age_bestsellers_previous_week, transactions, last_week)

    # construct the predictions
    customer_ids = customers['customer_id'].unique()
    sub = pd.DataFrame(
        {'customer_id': customer_encoder.inverse_transform(customer_ids),
         'prediction': ['' for _ in range(len(customer_ids))]})

    # add predictions for the customers and add most popular to them
    preds = []
    cid_2_age_bin = get_cid_2_age_bin(transactions)
    for customer_id in tqdm(customers['customer_id'].unique()):
        # if not in the age bin dict then use the garbage bin
        if customer_id not in cid_2_age_bin.keys():
            customer_age_bin = -1
        else:
            customer_age_bin = cid_2_age_bin.get(customer_id, -1)

        # return the customer specific predictions
        pred = c_id2predicted_article_ids.get(customer_id, [])
        # get the bestsellers of the age bin
        bestsellers_age_bin = bestsellers_age_bin_dict.get(customer_age_bin, [])

        # combine custom and bestseller predictions
        pred = pred + bestsellers_age_bin

        # only take the last predictions
        preds.append(pred[:12])

    # convert the predictions to a string
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub['prediction'] = preds

    # name and write the predictions
    sub.to_csv(f'../out/{sub_name}.csv.gz', index=False)


def make_and_write_predictions(model, test, X_test, bestsellers_last_week,
                               customers, customer_encoder,
                               sub_name):
    # get the predictions and convert to dict
    test['preds'] = model.predict(X_test)
    c_id2predicted_article_ids = get_cid_2_preds(test)

    # construct the predictions
    customer_ids = customers['customer_id'].unique()
    sub = pd.DataFrame(
        {'customer_id': customer_encoder.inverse_transform(customer_ids),
         'prediction': ['' for _ in range(len(customer_ids))]})

    # add predictions for the customers and add most popular to them
    preds = []
    for customer_id in tqdm(customers['customer_id'].unique()):
        # return the customer specific predictions
        pred = c_id2predicted_article_ids.get(customer_id, [])

        # combine custom and bestseller predictions
        pred = pred + bestsellers_last_week

        # only take the last predictions
        preds.append(pred[:12])

    # convert the predictions to a string
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub['prediction'] = preds
    # name and write the predictions
    sub.to_csv(f'../out/{sub_name}.csv.gz', index=False)


def age_smarter_bins():
    for smart_threshold in range(10, 200, 20):
        print(f'age smarter bins with threshold {smart_threshold}')
        articles, customers, transactions = read_data_set('feather')
        # articles, customers, transactions = part_data_set('01')

        # encode and decode customer ids
        customer_encoder = joblib.load('../data/customer_encoder.joblib')

        articles, customers, transactions = get_relevant_cols(articles,
                                                              customers,
                                                              transactions)

        last_week = 106
        test_week = last_week + 1

        # get transactions in the last n weeks
        transactions = last_n_week_transactions(transactions, last_week, 10)

        # label the ordered columns
        transactions['ordered'] = 1

        # combine the transactions dataframe with all the articles
        transactions = merge_transactions(transactions, articles, customers)
        del articles

        # change all the -1 ages to the most similar age using purchases
        transactions = change_age(transactions)

        # from here all the code has been used from unless marked otherwise
        # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
        # get the candidates last purchase
        # OWN: get the candidates age bin
        intervals = 'smart'
        candidates_age_bin, age_bestsellers_previous_week = \
            age_bin_candidates(transactions, test_week, intervals=intervals)

        # add the non-ordered items
        transactions = pd.concat([transactions, candidates_age_bin])
        transactions['ordered'] = transactions['ordered'].fillna(0)

        transactions = transactions.drop_duplicates(
            subset=['customer_id', 'product_type_no', 'week'], keep=False)

        transactions['ordered'] = transactions['ordered'].astype(np.uint8)

        # merge the previous week bestsellers to transactions as it adds the rank
        transactions = pd.merge(
            transactions,
            age_bestsellers_previous_week[
                ['week', 'age_bin', 'product_type_no',
                 'age_bestseller_rank']],
            on=['week', 'age_bin', 'product_type_no'],
            how='left'
        )

        # take only the transactions which aren't in the first week
        transactions = transactions[
            transactions['week'] != transactions['week'].min()]

        # change the values for the non-bestsellers to 999
        transactions['age_bestseller_rank'].fillna(999, inplace=True)

        # sort the transactions by week and customer_id
        transactions = transactions \
            .sort_values(['week', 'customer_id']) \
            .reset_index(drop=True)

        # take the training set as the transactions not happening in the test week
        train = transactions[transactions.week != test_week]
        # create the test week
        test = transactions[transactions.week == test_week] \
            .drop_duplicates(
            ['customer_id', 'product_type_no', 'sales_channel_id']) \
            .copy()

        # construct a suitable X and y (where y indicates ordered or not)
        X_train, y_train, = construct_X_y(train, usable_cols)
        X_test, _ = construct_X_y(test, usable_cols)

        # create a model
        group_sizes = get_group_sizes(train)
        model = lgb.LGBMRanker(objective='lambdarank',
                               metric='ndcg',
                               n_estimators=100,
                               importance_type='gain',
                               force_row_wise=True)
        model = model.fit(X=X_train, y=y_train, group=group_sizes)

        # show feature importances
        show_feature_importances(model)

        # make and write predictions
        sub_name = f'age_bin_prod_week_{last_week}_smart_threshold_{smart_threshold}'
        make_and_write_predictions_age(model, test, X_test,
                                       age_bestsellers_previous_week,
                                       customers, customer_encoder,
                                       transactions, last_week, sub_name)


def age_simple_bin():
    for last_week in [106, 105, 104, 103, 102]:
        for bin_size in [1, 2, 3, 4, 8, 16, 32, 64]:
            print(last_week, bin_size)
            articles, customers, transactions = read_data_set('feather')
            # articles, customers, transactions = part_data_set('01')

            # encode and decode customer ids
            customer_encoder = joblib.load('../data/customer_encoder.joblib')

            articles, customers, transactions = get_relevant_cols(articles,
                                                                  customers,
                                                                  transactions)

            test_week = last_week + 1

            # get transactions in the last n weeks
            transactions = last_n_week_transactions(transactions, last_week, 10)

            # label the ordered columns
            transactions['ordered'] = 1

            # combine the transactions dataframe with all the articles
            transactions = merge_transactions(transactions, articles, customers)
            del articles

            # change all the -1 ages to the most similar age using purchases
            transactions = change_age(transactions)

            # from here all the code has been used from unless marked otherwise
            # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
            # get the candidates last purchase
            # OWN: get the candidates age bin
            intervals = None
            candidates_age_bin, age_bestsellers_previous_week = \
                age_bin_candidates(transactions, test_week, bin_size,
                                   intervals=intervals)

            # add the non-ordered items
            transactions = pd.concat(
                [transactions, candidates_age_bin])
            transactions['ordered'] = transactions['ordered'].fillna(0)

            transactions = transactions.drop_duplicates(
                subset=['customer_id', 'product_type_no', 'week'], keep=False)

            transactions['ordered'] = transactions['ordered'].astype(np.uint8)

            # merge the previous week bestsellers to transactions as it adds the rank
            transactions = pd.merge(
                transactions,
                age_bestsellers_previous_week[
                    ['week', 'age_bin', 'product_type_no',
                     'age_bestseller_rank']],
                on=['week', 'age_bin', 'product_type_no'],
                how='left'
            )

            # take only the transactions which aren't in the first week
            transactions = transactions[
                transactions['week'] != transactions['week'].min()]

            # change the values for the non-bestsellers to 999
            transactions['age_bestseller_rank'].fillna(999, inplace=True)

            # sort the transactions by week and customer_id
            transactions = transactions \
                .sort_values(['week', 'customer_id']) \
                .reset_index(drop=True)

            # take the training set as the transactions not happening in the test week
            train = transactions[transactions.week != test_week]
            # create the test week
            test = transactions[transactions.week == test_week] \
                .drop_duplicates(
                ['customer_id', 'product_type_no', 'sales_channel_id']) \
                .copy()

            # construct a suitable X and y (where y indicates ordered or not)
            X_train, y_train, = construct_X_y(train, usable_cols)
            X_test, _ = construct_X_y(test, usable_cols)

            # create a model
            group_sizes = get_group_sizes(train)
            model = lgb.LGBMRanker(objective='lambdarank',
                                   metric='ndcg',
                                   n_estimators=100,
                                   importance_type='gain',
                                   force_row_wise=True)
            model = model.fit(X=X_train, y=y_train, group=group_sizes)

            # show feature importances
            show_feature_importances(model)

            # make and write predictions
            sub_name = f'age_bin_prod_week_{last_week}_bin_size_{bin_size}'
            make_and_write_predictions_age(model, test, X_test,
                                           age_bestsellers_previous_week,
                                           customers, customer_encoder,
                                           transactions, last_week, sub_name)


def just_popularity():
    articles, customers, transactions = read_data_set('feather')
    # articles, customers, transactions = part_data_set('01')

    # encode and decode customer ids
    customer_encoder = joblib.load('../data/customer_encoder.joblib')

    articles, customers, transactions = get_relevant_cols(articles,
                                                          customers,
                                                          transactions)

    last_week = 106
    test_week = 107

    # get transactions in the last n weeks
    transactions = last_n_week_transactions(transactions, last_week, 10)

    # label the ordered columns
    transactions['ordered'] = 1

    # combine the transactions dataframe with all the articles
    transactions = merge_transactions(transactions, articles, customers)
    del articles

    # from here all the code has been used from unless marked otherwise
    # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    # get the candidates last purchase
    candidates_bestsellers, bestsellers_previous_week = \
        get_candidate_bestsellers(transactions, test_week)
    # get the candidates bestsellers
    # candidates_last_purchase = last_purchase_candidates(transactions)
    # OWN: get the candidates age bin
    # bin_size = 2
    # intervals = None
    # intervals = [[16, 40], [41, 70]]
    # intervals = smart_merge_intervals(transactions)
    # candidates_age_bin, age_bestsellers_previous_week = \
    #     age_bin_candidates(transactions, bin_size, intervals=intervals)

    # add the non-ordered items
    transactions = pd.concat(
        [transactions, candidates_bestsellers])
    transactions['ordered'] = transactions['ordered'].fillna(0)

    transactions = transactions.drop_duplicates(
        subset=['customer_id', 'article_id', 'week'], keep=False)

    transactions['ordered'] = transactions['ordered'].astype(
        np.uint8)
    # merge the previous week bestsellers to transactions as it adds the rank
    # transactions = pd.merge(
    #     transactions,
    #     age_bestsellers_previous_week[
    #         ['week', 'age_bin', 'article_id', 'age_bestseller_rank']],
    #     on=['week', 'age_bin', 'article_id'],
    #     how='left'
    # )
    transactions = pd.merge(
        transactions,
        bestsellers_previous_week[
            ['week', 'article_id', 'bestseller_rank']],
        on=['week', 'article_id'],
        how='left'
    )

    # take only the transactions which aren't in the first week
    transactions = transactions[
        transactions['week'] != transactions['week'].min()]

    # change the values for the non-bestsellers to 999
    # transactions['age_bestseller_rank'].fillna(999, inplace=True)
    transactions['bestseller_rank'].fillna(999, inplace=True)

    # sort the transactions by week and customer_id
    transactions = transactions \
        .sort_values(['week', 'customer_id']) \
        .reset_index(drop=True)

    # take the training set as the transactions not happening in the test week
    train = transactions[transactions.week != test_week]
    # create the test week
    test = transactions[transactions.week == test_week] \
        .drop_duplicates(
        ['customer_id', 'article_id', 'sales_channel_id']) \
        .copy()

    # construct a suitable X and y (where y indicates ordered or not)
    X_train, y_train, = construct_X_y(train, usable_cols)
    X_test, _ = construct_X_y(test, usable_cols)

    # create a model
    group_sizes = get_group_sizes(train)
    model = lgb.LGBMRanker(objective='lambdarank',
                           metric='ndcg',
                           n_estimators=100,
                           importance_type='gain',
                           force_row_wise=True)
    model = model.fit(X=X_train, y=y_train, group=group_sizes)

    # show feature importances
    show_feature_importances(model)

    # bestsellers last week
    bestsellers_last_week = \
        bestsellers_previous_week[
            bestsellers_previous_week['week'] == test_week][
            'article_id'].tolist()

    # make and write predictions
    sub_name = f'simply_popularity_week_{last_week}'
    make_and_write_predictions(model, test, X_test, bestsellers_last_week,
                               customers, customer_encoder, sub_name)


if __name__ == '__main__':
    age_smarter_bins()
    # age_simple_bin()
    # just_popularity()
    # prepare_feather_datasets()
    # for last_week in [106, 105, 104, 103, 102]:
    #     for bin_size in [1, 2, 3, 4, 8, 16, 32, 64]:
    #         print(last_week, bin_size)
    #         articles, customers, transactions = read_data_set('feather')
    #         # articles, customers, transactions = part_data_set('01')
    #
    #         # encode and decode customer ids
    #         customer_encoder = joblib.load('../data/customer_encoder.joblib')
    #
    #         articles, customers, transactions = get_relevant_cols(articles,
    #                                                               customers,
    #                                                               transactions)
    #
    #         test_week = last_week + 1
    #
    #         # get transactions in the last n weeks
    #         transactions = last_n_week_transactions(transactions, 10)
    #
    #         # label the ordered columns
    #         transactions['ordered'] = 1
    #
    #         # combine the transactions dataframe with all the articles
    #         transactions = merge_transactions(transactions, articles, customers)
    #         del articles
    #
    #         # from here all the code has been used from unless marked otherwise
    #         # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    #         # get the candidates last purchase
    #         # candidates_bestsellers, bestsellers_previous_week = \
    #         #     get_candidate_bestsellers(transactions)
    #         # get the candidates bestsellers
    #         # candidates_last_purchase = last_purchase_candidates(transactions)
    #         # OWN: get the candidates age bin
    #         # bin_size = 2
    #         intervals = None
    #         # intervals = [[16, 40], [41, 70]]
    #         # intervals = smart_merge_intervals(transactions)
    #         candidates_age_bin, age_bestsellers_previous_week = \
    #             age_bin_candidates(transactions, bin_size, intervals=intervals)
    #
    #         # add the non-ordered items
    #         transactions = pd.concat(
    #             [transactions, candidates_age_bin])
    #         transactions['ordered'] = transactions['ordered'].fillna(0)
    #
    #         transactions = transactions.drop_duplicates(
    #             subset=['customer_id', 'article_id', 'week'], keep=False)
    #
    #         transactions['ordered'] = transactions['ordered'].astype(
    #             np.uint8)
    #         # merge the previous week bestsellers to transactions as it adds the rank
    #         transactions = pd.merge(
    #             transactions,
    #             age_bestsellers_previous_week[
    #                 ['week', 'age_bin', 'article_id', 'age_bestseller_rank']],
    #             on=['week', 'age_bin', 'article_id'],
    #             how='left'
    #         )
    #         # transactions = pd.merge(
    #         #     transactions,
    #         #     bestsellers_previous_week[
    #         #         ['week', 'article_id', 'bestseller_rank']],
    #         #     on=['week', 'article_id'],
    #         #     how='left'
    #         # )
    #
    #         # take only the transactions which aren't in the first week
    #         transactions = transactions[
    #             transactions['week'] != transactions['week'].min()]
    #
    #         # change the values for the non-bestsellers to 999
    #         transactions['age_bestseller_rank'].fillna(999, inplace=True)
    #         # transactions['bestseller_rank'].fillna(999, inplace=True)
    #
    #         # sort the transactions by week and customer_id
    #         transactions = transactions \
    #             .sort_values(['week', 'customer_id']) \
    #             .reset_index(drop=True)
    #
    #         # take the training set as the transactions not happening in the test week
    #         train = transactions[transactions.week != test_week]
    #         # create the test week
    #         test = transactions[transactions.week == test_week] \
    #             .drop_duplicates(
    #             ['customer_id', 'article_id', 'sales_channel_id']) \
    #             .copy()
    #
    #         # construct a suitable X and y (where y indicates ordered or not)
    #         X_train, y_train, = construct_X_y(train, usable_cols)
    #         X_test, _ = construct_X_y(test, usable_cols)
    #
    #         # create a model
    #         group_sizes = get_group_sizes(train)
    #         model = lgb.LGBMRanker(objective='lambdarank',
    #                                metric='ndcg',
    #                                n_estimators=100,
    #                                importance_type='gain',
    #                                force_row_wise=True)
    #         model = model.fit(X=X_train, y=y_train, group=group_sizes)
    #
    #         # show feature importances
    #         show_feature_importances()
    #         # bestsellers last week
    #         # bestsellers_last_week = \
    #         #     age_bestsellers_previous_week[
    #         #         age_bestsellers_previous_week['week'] == test_week][
    #         #         'article_id'].tolist()
    #
    #         # make and write predictions
    #         sub_name = f'age_bin_week_{last_week}_bin_size_{bin_size}'
    #         make_and_write_predictions_age(model, test,
    #                                        age_bestsellers_previous_week,
    #                                        customers, customer_encoder,
    #                                        transactions, last_week, sub_name)
