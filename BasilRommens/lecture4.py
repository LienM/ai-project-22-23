import collections

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

rand = 64

obj = "class"  # "class" or "rank"
N = 15000
N_div = 20
n_iter = 2  # num of iteration
idx_file = "exp05"
len_hist = 366
n_round = 4000
n_splits = 1
tmp_top = 200
tr_set = [1, 8, 15, 22]  # set of train date
len_tr = 7  # length of validation period
nobuy = 20  # num of negative samples


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
def get_candidate_bestsellers(transactions):
    # get the average price of each article in a given week
    mean_price = transactions \
        .groupby(['week', 'article_id'])['price'].mean()

    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='dense', ascending=False) \
        .groupby('week').head(20).rename('bestseller_rank').astype('int8')

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


def age_bin_candidates(transactions, bin_size=1):
    # add bin size to the age column
    transactions['age_bin'] = transactions['age'] // bin_size
    # TRAINING WEEK CANINATES
    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    bestsellers_previous_week = transactions \
        .groupby(['week', 'age_bin'])['article_id'].value_counts() \
        .groupby(['week', 'age_bin']).rank(method='dense', ascending=False) \
        .groupby(['week', 'age_bin']).head(12).rename('age_bestseller_rank') \
        .astype('uint8') \
        .reset_index()
    bestsellers_previous_week.week += 1  # move the week 1 week forward
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id']) \
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


def get_last_weeks_popularity(pop_type, transactions):
    if pop_type == 'item_popularity':
        transactions_up_shift_week = transactions.copy()
        transactions_up_shift_week['week'] += 1
        pop = transactions_up_shift_week.groupby(['week', 'article_id'],
                                                 as_index=False)[
            ['ordered']].sum()
        pop[pop_type] = pop['ordered']
        pop = pop.drop('ordered', axis=1)
        transactions = transactions.merge(pop, on=['week', 'article_id'])
    elif pop_type == 'colour_popularity':
        transactions_up_shift_week = transactions.copy()
        transactions_up_shift_week['week'] += 1
        pop = transactions_up_shift_week.groupby(['week', 'colour_group_code'],
                                                 as_index=False)[
            ['ordered']].sum()
        pop[pop_type] = pop['ordered']
        pop = pop.drop('ordered', axis=1)
        transactions = transactions.merge(pop, on=['week', 'colour_group_code'])
    elif pop_type == 'product_type_popularity':
        transactions_up_shift_week = transactions.copy()
        transactions_up_shift_week['week'] += 1
        pop = transactions_up_shift_week.groupby(['week', 'product_type_no'],
                                                 as_index=False)[
            ['ordered']].sum()
        pop[pop_type] = pop['ordered']
        pop = pop.drop('ordered', axis=1)
        transactions = transactions.merge(pop, on=['week', 'product_type_no'])

    return transactions[pop_type]


if __name__ == '__main__':
    # prepare_feather_datasets()
    articles, customers, transactions = read_data_set('feather')
    # articles, customers, transactions = part_data_set('01')
    customer_encoder = joblib.load('../data/customer_encoder.joblib')

    # taking the relevant columns of all dataframes
    relevant_article_cols = ['article_id', 'product_type_no',
                             'graphical_appearance_no', 'colour_group_code',
                             'perceived_colour_value_id',
                             'perceived_colour_master_id', 'department_no',
                             'index_group_no', 'section_no', 'garment_group_no']
    relevant_customer_cols = ['customer_id', 'FN', 'Active',
                              'fashion_news_frequency', 'age', 'postal_code']
    relevant_transaction_cols = ['t_dat', 'customer_id', 'article_id', 'price',
                                 'sales_channel_id']
    articles = articles[relevant_article_cols]
    customers = customers[relevant_customer_cols]
    transactions = transactions[relevant_transaction_cols]

    # data set construction by taking only last week of transactions
    transactions['week'] = transactions['t_dat'].dt.isocalendar().year * 53 + \
                           transactions['t_dat'].dt.isocalendar().week
    transactions['week'] = rankdata(transactions['week'], 'dense')

    test_week = transactions.week.max() + 1
    transactions = transactions[
        transactions.week > transactions.week.max() - 10]

    # label the ordered columns
    transactions['ordered'] = 1

    # Concat the negative samples to the positive samples:
    # transactions = pd.concat([transactions, neg_transactions])
    transactions = transactions.merge(articles, on='article_id')
    del articles
    transactions = transactions.merge(customers, on='customer_id')
    transactions = transactions.reset_index(drop=True)

    # from here all the code has been used from
    # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    # unless marked otherwise
    # get the candidates last purchase
    # candidates_bestsellers, bestsellers_previous_week = \
    #     get_candidate_bestsellers(transactions)
    # get the candidates bestsellers
    # candidates_last_purchase = last_purchase_candidates(transactions)
    # own: get the candidates age bin
    bin_size = 16
    candidates_age_bin, age_bestsellers_previous_week = age_bin_candidates(
        transactions, bin_size)

    # add the non-ordered items
    transactions = pd.concat(
        [transactions, candidates_age_bin])
    transactions['ordered'] = transactions['ordered'].fillna(0)

    transactions = transactions.drop_duplicates(
        subset=['customer_id', 'article_id', 'week'], keep=False)

    # Concat the negative samples to the positive samples:
    # transactions = transactions.merge(articles, on='article_id')
    # del articles
    # transactions = transactions.merge(customers, on='customer_id')
    # del customers
    # transactions = transactions.reset_index(drop=True)

    last_transaction_week = transactions['week'].max()
    # # add last weeks item popularity to transactions
    # pop_type = 'item_popularity'
    # transactions[pop_type] = get_last_weeks_popularity(pop_type, transactions)
    # # # add last weeks colour popularity to transactions
    # pop_type = 'colour_popularity'
    # transactions[pop_type] = get_last_weeks_popularity(pop_type, transactions)
    # # # add last weeks product type popularity to transactions
    # pop_type = 'product_type_popularity'
    # transactions[pop_type] = get_last_weeks_popularity(pop_type, transactions)

    usable_cols = ['article_id', 'product_type_no', 'graphical_appearance_no',
                   'perceived_colour_value_id', 'perceived_colour_master_id',
                   'department_no', 'index_group_no', 'section_no',
                   'FN', 'Active', 'fashion_news_frequency', 'age',
                   'age_bestseller_rank']
    transactions['ordered'] = transactions['ordered'].astype(
        np.uint8)
    # merge the previous week bestsellers to the transactions as it will add the
    # rank
    transactions = pd.merge(
        transactions,
        age_bestsellers_previous_week[
            ['week', 'age_bin', 'article_id', 'age_bestseller_rank']],
        on=['week', 'age_bin', 'article_id'],
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
    group_sizes = get_group_sizes(train)
    # create the test week
    test = transactions[transactions.week == test_week] \
        .drop_duplicates(['customer_id', 'article_id', 'sales_channel_id']) \
        .copy()

    # construct a suitable X and y (where y indicates ordered or not)
    X_train = train[usable_cols]
    y_train = train[['ordered']]
    X_test = test[usable_cols]

    # create a model
    model = lgb.LGBMRanker(objective='lambdarank',
                           metric='ndcg',
                           n_estimators=100,
                           importance_type='gain',
                           force_row_wise=True)
    model = model.fit(X=X_train, y=y_train, group=group_sizes,
                      eval_metric='ndcg', eval_set=[(X_train, y_train)],
                      eval_group=[group_sizes])

    for i in model.feature_importances_.argsort()[::-1]:
        print(usable_cols[i], model.feature_importances_[
            i] / model.feature_importances_.sum())

    # get the predictions
    test['preds'] = model.predict(X_test)

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()

    bestsellers_last_week = \
        age_bestsellers_previous_week[
            age_bestsellers_previous_week.week == age_bestsellers_previous_week.week.max()][
            'article_id'].tolist()

    # construct the predictions
    customer_ids = customers['customer_id'].unique()
    sub = pd.DataFrame(
        {'customer_id': customer_encoder.inverse_transform(customer_ids),
         'prediction': ['' for _ in range(len(customer_ids))]})

    # add predictions for the customers and add most popular to them
    preds = []
    for customer_id in customers['customer_id'].unique():
        pred = c_id2predicted_article_ids.get(customer_id, [])
        # pred = pred + bestsellers_last_week
        preds.append(pred[:12])

    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub['prediction'] = preds

    sub_name = 'age_bin_16'
    sub.to_csv(f'../data/{sub_name}.csv.gz', index=False)

    # the commented out code has never been used
    # accuracy = model.score(X_va, y_va)
    # print(accuracy)
    # reader = Reader(rating_scale=(0, 1))
    # data = Dataset.load_from_df(
    #     X_reduced[["customer_id", "article_id", "relevant"]], reader)
    # trainset = data.build_full_trainset()
    # # Use the famous SVD algorithm.
    # model = SVD()
    # model.fit(trainset)

    # topn = predict_topn(X_reduced, model)

    # # train predictor
    # # get most popular items
    # top_counts = collections.Counter(X_transactions.groupby('article_id')[
    #                                      'article_id'].count().to_dict()).most_common(
    #     12)

    # write all the predictions
    # predictions = ['customer_id,prediction\n']
    # for customer_id in customers['customer_id'].unique():
    #     if customer_id not in topn.keys():
    #         continue
    #     top_counts = topn[customer_id]
    #     customer_predictions = popular_predictor(customer_id, top_counts)
    #     prediction_line = customer_id + ',' + customer_predictions + '\n'
    #     predictions.append(prediction_line)
    #
    # with open('../data/predictions.csv', 'w') as f:
    #     f.writelines(predictions)
    # pred = pd.read_csv('../data/predictions.csv')

    # print(map_at_k(rel_items=y_transactions, users=pred, n=12, k=12))
