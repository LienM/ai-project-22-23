import numpy as np
import pandas as pd
import random
from scipy import spatial
import lightgbm as lgb

from ast import literal_eval

from LabelEncoder import MultiLabelEncoder
from utils import *
from Parameters import PARAM

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


multi_encoder = MultiLabelEncoder()


def load_data():
    print("loading articles ...")
    articles = pd.read_csv('../../articles_sample1.csv.gz')
    print("loading customers ...")
    customers = pd.read_csv('../../customers_sample1.csv.gz')
    print("loading transactions ...")
    transactions = pd.read_csv('../../transactions_sample1.csv.gz')
    return transactions, articles, customers


def load_pkl():
    print("loading articles ...")
    articles = pd.read_pickle('articles_pp.pkl')
    print("loading customers ...")
    customers = pd.read_pickle('customers_pp.pkl')
    print("loading transactions ...")
    transactions = pd.read_pickle('transactions_pp.pkl')
    print("loading samples ...")
    samples = pd.read_pickle('samples.pkl')
    return transactions, articles, customers, samples


def create_sample(transactions, articles, customers):
    print("creating samples ...")
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
    samples.to_pickle("samples.pkl")
    return samples


def preprocess_transactions(transactions):
    print("preprocessing transactions")
    transactions_processed = transactions[PARAM["transaction"]].copy()
    transactions_processed.head()

    transactions_processed = transactions_processed.fillna(0)

    multi_encoder.create_encoder("article_id")
    multi_encoder.create_encoder("customer_id")

    transactions_processed = multi_encoder.encode("article_id", "article_id", transactions_processed)
    transactions_processed = multi_encoder.encode("customer_id", "customer_id", transactions_processed)
    transactions_processed.to_pickle("transactions_pp.pkl")
    return transactions_processed


def preprocess_articles(articles):
    print("preprocessing articles")
    articles_processed = articles[PARAM["article"]].copy()
    articles_processed = multi_encoder.encode("article_id", "article_id", articles_processed)
    articles_processed = articles_processed.fillna(0)

    # materials
    materials = scrape_materials()
    articles_processed = extract_article_material(articles_processed, materials)
    multi_encoder.create_encoder("material")
    articles_processed = multi_encoder.encode("material", "material", articles_processed)
    articles_processed.to_pickle("articles_pp.pkl")

    return articles_processed


def preprocess_customers(customers):
    print("preprocessing customers")
    customers_processed = customers[PARAM["customer"]].copy()
    multi_encoder.create_encoder("postal_code")
    customers_processed = multi_encoder.encode("customer_id", "customer_id", customers_processed)
    customers_processed = multi_encoder.encode("postal_code", "postal_code", customers_processed)
    customers_processed = customers_processed.fillna(0)
    customers_processed.to_pickle("customers_pp.pkl")
    return customers_processed


def preprocess_data():
    transactions, articles, customers = load_data()
    transactions = preprocess_transactions(transactions)
    articles = preprocess_articles(articles)
    customers = preprocess_customers(customers)
    samples = create_sample(transactions, articles, customers)

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
    # recent_items = recent_items.sort_values(by="count", ascending=False)["article_id"].head(100)
    recent_items = recent_items.sort_values(by="count", ascending=False).head(1000)
    recent_items.drop(columns=["count"], inplace=True)
    recent_items = recent_items.merge(articles, how="inner", on="article_id")
    # return list(recent_items)

    return recent_items


def candidates_selection(transactions, articles, customers, samples, candidates=None):
    recent_items = recent_sales_k_months(samples, articles, 2)
    print("candidate selection ...")
    candidates = k_nn_clustering(transactions, recent_items, customers, candidates)

    return candidates


def k_nn_clustering(transactions, articles, customers, candidates=None, k=13673):
    # prepare transaction data
    sales = transactions.merge(articles, how="inner", on='article_id')
    sales.drop(columns=["sales_channel_id", "t_dat"], inplace=True)
    articles = articles.merge(transactions[["article_id", "price"]].copy(), how="inner", on="article_id")
    # prepare candidates dict
    if candidates is None:
        candidates = {"customers": [], "sims": []}
    c = list(customers["customer_id"].values)
    i = k
    for customer in range(k, len(c)):
        if i % 50 == 0:
            if i != k:
                print(f"{i}/{len(c)}")
                save_candidates(candidates)
                candidates = {"customers": [], "sims": []}
        # check if customer is already present in candidates dict
        # if customer not in candidates:
        #     candidates[customer] = []
        # similarities = pd.DataFrame(columns=['article_id', 'sim'])
        similarities = {}

        # filter purchases of given customer
        purchases = sales[sales["customer_id"] == c[customer]].copy()
        # purchased_articles = purchases["article_id"].values

        # skip if customer has not made purchases yet
        if purchases.shape[0] == 0:
            continue

        purchases = purchases.drop(columns=["customer_id", "article_id"])


        k = 5
        if purchases.shape[0] <= 5:
            k = 1

        # find clusters and calculate centroids
        clusters = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(purchases)
        centroids = clusters.cluster_centers_
        added_articles = []

        # find K nearest neighbourgs
        for index, article in articles.iterrows():
            article_id = int(article["article_id"])
            if article_id in added_articles:
                continue

            # if article_id in purchased_articles:
            #     continue
            value_list = article.values.tolist()
            value_list.pop(0)

            # calculate similarities between clusters and item
            top_similarity = 0
            for centroid in centroids:
                similarity = 1 - spatial.distance.cosine(value_list, centroid)

                # if similarity with a cluster is high enough, item gets added to candidates
                if similarity >= top_similarity:
                    top_similarity = similarity
            similarities[index] = {"article_id": article_id, "sim": top_similarity}
            added_articles.append(article_id)

        sim_df = pd.DataFrame.from_dict(similarities, "index")
        sim_df = sim_df.sort_values(by=["sim"], ascending=False)
        candidates["customers"].append(c[customer])
        candidates["sims"].append(sim_df["article_id"].head(200).values.tolist())
        i += 1
    return candidates


def rank_candidates(candidates, articles, samples):
    samples = samples[PARAM["samples"]].copy()
    samples["customer_id"] = pd.to_numeric(samples["customer_id"])
    samples["article_id"] = pd.to_numeric(samples["article_id"])
    samples["ordered"] = pd.to_numeric(samples["ordered"])

    # setting up data
    training_data = extract_all_but_last_week_sales(samples, False)
    validation_data = extract_last_week_sales(samples)

    qids_train = training_data.groupby(["customer_id"])["article_id"].count().to_numpy()
    qids_validation = validation_data.groupby(["customer_id"])["article_id"].count().to_numpy()

    training_data = training_data.drop(["t_dat"], axis=1)
    x_train = training_data.drop(columns=["ordered"])
    y_train = training_data["ordered"]

    validation_data = validation_data.drop(["t_dat"], axis=1)
    x_test = validation_data.drop(columns=["ordered"])
    y_test = validation_data["ordered"]

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="auc",
        boosting_type="dart",
        n_estimators=100,
        importance_type='gain',
        verbose=10,
        min_child_samples=1,
        num_leaves=50
    )

    ranker.fit(
        X=x_train,
        y=y_train,
        group=qids_train,
        eval_set=[(x_test, y_test)],
        eval_group=[qids_validation],
        eval_at=10,
    )

    candidates["prediction"] = []
    for customer in range(len(candidates["customers"])):
        candidate_articles = articles.loc[articles['article_id'].isin(candidates["sims"][customer])].copy()
        candidate_articles['customer_id'] = candidates["customers"][customer]
        if candidate_articles.shape[0] > 0:
            candidate_articles["prediction"] = ranker.predict(candidate_articles)
            candidate_articles = candidate_articles.sort_values(by=["prediction"], ascending=False)
            candidates["prediction"].append(list(candidate_articles["article_id"].head(12)))

    return candidates


def save_candidates(candidates):
    df = pd.DataFrame(candidates)
    df.to_csv('candidates1.csv', mode='a', index=False, header=False)


def load_candidates(file="candidates1.csv"):
    candidates = pd.read_csv(file)
    candidates["sims"] = candidates["sims"].apply(literal_eval)
    candidates = candidates.to_dict(orient='list')
    return candidates


def create_submission(candidates):
    # create dataframe
    submission = pd.DataFrame(columns=["customer_id", "prediction"])
    submission["customer_id"] = candidates["customers"]
    # submission["prediction"] = candidates["prediction"]

    # label decode
    submission = multi_encoder.decode("customer_id", "customer_id", submission)
    submission = submission.reset_index()
    for prediction in range(len(candidates["prediction"])):
        dataframe = pd.DataFrame(columns=["article_id"])
        dataframe["article_id"] = candidates["prediction"][prediction]
        dataframe = multi_encoder.decode("article_id", "article_id", dataframe)
        submission.at[prediction, 'prediction'] = list(dataframe["article_id"])

    submission.drop(columns=["index"], inplace=True)
    submission.to_csv('submission.csv', index=False, header=True)


if __name__ == "__main__":
    transactions, articles, customers, samples = None, None, None, None
    # load dataset
    if PARAM["PP"]:
        transactions, articles, customers, samples = preprocess_data()
    else:
        transactions, articles, customers, samples = load_pkl()


    candidates = {}
    # run candidate selection
    if PARAM["SELECT"]:
        candidates = candidates_selection(transactions, articles, customers, samples)

    # save generated candidates
    if PARAM["SAVE"]:
        save_candidates(candidates)

    # load candidates from file
    if PARAM["LOAD"]:
        candidates = load_candidates()



    # rank candidates and calculate precision
    # popular_items = recent_sales_k_months(transactions, articles, 6)
    # for customer in range(len(candidates["customers"])):
    #     candidates["sims"][customer] += popular_items

    candidates = rank_candidates(candidates, articles, samples)
    map_at_12(samples, candidates)

    # write submission
    if PARAM["SUBMIT"]:
        create_submission(candidates)

