import numpy as np
import pandas as pd
import random
from scipy import spatial
from LabelEncoder import MultiLabelEncoder
import lightgbm as lgb
import csv

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


multi_encoder = MultiLabelEncoder()
WRITE = True


def load_data():
    print("loading articles ...")
    articles = pd.read_csv('../../articles_sample1.csv.gz')
    print("loading customers ...")
    customers = pd.read_csv('../../customers_sample1.csv.gz')
    print("loading transactions ...")
    transactions = pd.read_csv('../../transactions_sample1.csv.gz')
    print("creating samples ...")
    samples = create_sample(transactions, articles, customers)
    return transactions, articles, customers, samples


def create_sample(transactions, articles, customers):
    transactions['ordered'] = 1
    positive_pairs = list(map(tuple, transactions[['customer_id', 'article_id']].drop_duplicates().values))
    # Extract real values
    real_dates = transactions["t_dat"].unique()
    real_customers = transactions["customer_id"].unique()
    real_articles = transactions["article_id"].unique()
    real_channels = transactions["sales_channel_id"].unique()
    article_and_price = transactions[["article_id", "price"]].drop_duplicates("article_id").set_index(
        "article_id").squeeze()
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
    ordered = np.array([0] * num_neg_samples)
    # Assign to every article a real price.
    neg_prices = article_and_price[neg_articles].values
    neg_transactions = pd.DataFrame([neg_dates, neg_customers, neg_articles, neg_prices, neg_channels, ordered],
                                    index=transactions.columns).T
    # Remove random negative samples that actually coincide with positives
    df = neg_transactions[
        ~neg_transactions.set_index(["customer_id", "article_id"]).index.isin(positive_pairs)
    ]

    # Remove any excess
    chosen_neg_transactions = df.sample(num_neg_pos)
    # Concat the negative samples to the positive samples:
    samples = pd.concat([transactions, chosen_neg_transactions])
    samples = samples.merge(customers, how="inner", on='customer_id')
    samples = samples.merge(articles, how="inner", on='article_id')
    return samples


def preprocess_transactions(transactions):
    # I'm dropping a lot of columns, use them in your engineering tasks!
    transactions_processed = transactions[['customer_id', 'article_id', 'sales_channel_id', 'price']].copy()
    transactions_processed.head()

    transactions_processed = transactions_processed.fillna(0)

    multi_encoder.create_encoder("article_id")
    multi_encoder.create_encoder("customer_id")

    transactions_processed = multi_encoder.encode("article_id", "article_id", transactions_processed)
    transactions_processed = multi_encoder.encode("customer_id", "customer_id", transactions_processed)

    return transactions_processed


def preprocess_articles(articles):
    articles_processed = articles[
        ["article_id", "colour_group_code", "department_no", "index_group_no", "section_no", "garment_group_no"]].copy()
    articles_processed = multi_encoder.encode("article_id", "article_id", articles_processed)
    articles_processed = articles_processed.fillna(0)

    return articles_processed


def preprocess_customers(customers):
    customers_processed = customers[["customer_id", "age", "postal_code"]].copy()
    multi_encoder.create_encoder("postal_code")
    customers_processed = multi_encoder.encode("customer_id", "customer_id", customers_processed)
    customers_processed = multi_encoder.encode("postal_code", "postal_code", customers_processed)
    customers_processed = customers_processed.fillna(0)

    return customers_processed


def preprocess_data():
    transactions, articles, customers, samples = load_data()

    transactions = preprocess_transactions(transactions)
    articles = preprocess_articles(articles)
    customers = preprocess_customers(customers)

    return transactions, articles, customers, samples


def recent_sales_k_months(transactions, articles, k, candidates=None):
    # Calculate age of transactions in months
    if candidates is None:
        candidates = {}
    dated_transactions = transactions.loc[transactions["ordered"] == 1][["article_id", "t_dat"]]
    dated_transactions["t_dat"] = pd.to_datetime(dated_transactions["t_dat"])
    latest_date = dated_transactions["t_dat"].max()
    dated_transactions["transaction_age_months"] = 12 * (latest_date.year - dated_transactions["t_dat"].dt.year) + (
                latest_date.month - dated_transactions["t_dat"].dt.month)

    # Getting all sold items within the last 6 months
    recent_items = dated_transactions.loc[dated_transactions["transaction_age_months"] < k]
    recent_items = recent_items["article_id"].value_counts().rename_axis("article_id").reset_index(name="count")

    recent_items = recent_items.merge(articles, how="inner", on="article_id")
    # # Setting up dict with all candidates for
    # for customer_id in customers["customer_id"].values:
    #     candidates[customer_id] = recent_items["article_id"].head(100)
    return recent_items


def candidates_selection(transactions, articles, customers, samples):
    # recent_items = recent_sales_k_months(samples, articles, 6)
    print("starting candidate selection")
    candidates = k_kk_clustering(transactions, articles, customers)
    if WRITE:
        save_candidates(candidates)
    # candidates = rank_candidates(candidates, articles, samples)
    return candidates


def k_kk_clustering(transactions, articles, customers, candidates=None):
    # prepare transaction data
    sales = transactions.merge(articles, how="inner", on='article_id')
    # prepare candidates dict
    if candidates is None:
        candidates = {}
    c = customers["customer_id"].values
    for customer in c:
        # check if customer is already present in candidates dict
        if customer not in candidates:
            candidates[customer] = []
        similarities = pd.DataFrame(columns=['article_id', 'sim'])

        # filter purchases of given customer
        purchases = sales.loc[sales["customer_id"] == customer].copy()
        purchased_articles = purchases["article_id"].values
        purchases.drop(columns=["customer_id", "article_id", "sales_channel_id", "price"], inplace=True)

        # skip if customer has not made purchases yet
        if purchases.shape[0] == 0:
            continue
        k = 6
        if purchases.shape[0] <= 12:
            k = 2
            if purchases.shape[0] == 1:
                k = 1

        # find clusters and calculate centroids
        clusters = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(purchases)
        centroids = clusters.cluster_centers_

        # find K nearest neighbourgs
        for index, article in articles.iterrows():
            article_id = int(article["article_id"])
            if article_id in purchased_articles:
                continue
            value_list = article.values.flatten().tolist()
            value_list.pop(0)
            # calculate similarities between clusters and item
            top_similarity = 0
            for centroid in centroids:
                similarity = 1 - spatial.distance.cosine(value_list, centroid)
                # if similarity with a cluster is high enough, item gets added to candidates
                if similarity >= top_similarity:
                    top_similarity = similarity
            similarities.loc[len(similarities.index)] = [article_id, top_similarity]
        similarities = similarities.sort_values(by=["sim"], ascending=False)
        candidates[customer] = similarities["article_id"].head(200)

    return candidates


def rank_candidates(candidates, articles, samples):
    # setting up data
    X_train, X_test, y_train, y_test = train_test_split(samples.drop('ordered', axis=1),
                                                        samples['ordered'], test_size=0.10,
                                                        random_state=42)
    training_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)

    # setting up training parameters
    param = {'num_leaves': 31, 'objective': 'binary', 'metric': 'auc'}
    num_round = 10
    bst = lgb.train(param, training_data, num_round, valid_sets=[validation_data])
    for customer in candidates.keys():
        candidate_articles = articles.loc[articles['article_id'].isin(candidates[customer])].copy()
        if candidate_articles.shape[0] > 0:
            predictions = bst.predict(candidate_articles)
            candidate_articles = candidate_articles.sort_values(by=["prediction"], ascending=False)
            candidates[customer] = candidate_articles.head(12)


def save_candidates(candidates):
    with open('candidates.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=candidates.keys())
        writer.writeheader()
        writer.writerow(candidates)

if __name__ == "__main__":
    transactions, articles, customers, samples = preprocess_data()
    candidates = candidates_selection(transactions, articles, customers, samples)

