import numpy as np
import pandas as pd
import random
from scipy import spatial
import lightgbm as lgb
from sklearn.cluster import KMeans

from ast import literal_eval
from LabelEncoder import MultiLabelEncoder
from utils import *
from Parameters import PARAM, FIXED_PARAMS, SEARCH_PARAMS
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split


multi_encoder = MultiLabelEncoder()


def load_data():
    print("loading articles ...")
    articles = pd.read_csv('../data/articles_sample1.csv.gz')
    print("loading customers ...")
    customers = pd.read_csv('../data/customers_sample1.csv.gz')
    print("loading transactions ...")
    transactions = pd.read_csv('../data/transactions_sample1.csv.gz')
    return transactions, articles, customers


def load_pkl():
    print("loading articles ...")
    articles = pd.read_pickle('../articles_pp.pkl')
    print("loading customers ...")
    customers = pd.read_pickle('../customers_pp.pkl')
    print("loading transactions ...")
    transactions = pd.read_pickle('../transactions_pp.pkl')
    print("loading samples ...")
    samples = pd.read_pickle('../samples.pkl')
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

    random.seed(42)

    # Afterwards, we need to remove potential duplicates, so we'll sample too many.
    num_neg_samples = int(num_neg_pos * 1.1)

    num_popular = int(num_neg_samples)
    # popularity based negative sampling
    popular_items = popular_articles(transactions, articles)
    active_buyers = most_active_customers(transactions)


    popular_neg_dates = np.random.choice(real_dates, size=num_popular)
    popular_neg_articles = popular_items["article_id"].sample(n=num_popular, weights=popular_items["s_weight"], replace=True, random_state=42)
    popular_neg_channels = np.random.choice(real_channels, size=num_popular)
    popular_ordered = np.array([0] * num_popular)
    popular_neg_customers = active_buyers["customer_id"].sample(n=num_popular, weights=active_buyers["s_weight"], replace=True, random_state=42)
    popular_article_and_price = popular_items[["article_id", "price"]].set_index(
        "article_id").squeeze()
    popular_neg_prices = popular_article_and_price[popular_neg_articles].values

    indexes = pd.DataFrame({'index': np.arange(num_popular)})
    popular_neg_transactions = pd.DataFrame([np.arange(num_popular),popular_neg_customers, popular_neg_articles, popular_neg_channels, popular_neg_prices, popular_neg_dates, popular_ordered],
                                            index=["index", "customer_id", "article_id", "sales_channel_id", "price", "t_dat", "ordered"]).T
    popular_neg_transactions = popular_neg_transactions[["customer_id", "article_id", "sales_channel_id", "price", "t_dat", "ordered"]]

    # random negative sampling
    # num_random = int(num_neg_samples/3
    num_random = 0
    random_neg_dates = np.random.choice(real_dates, size=num_random)
    random_neg_articles = np.random.choice(real_articles, size=num_random)
    random_neg_customers = np.random.choice(real_customers, size=num_random)
    random_neg_channels = np.random.choice(real_channels, size=num_random)
    random_ordered = np.array([0] * num_random)
    # Assign to every article a real price.
    random_neg_prices = article_and_price[random_neg_articles].values
    random_neg_transactions = pd.DataFrame([random_neg_customers, random_neg_articles, random_neg_channels, random_neg_prices, random_neg_dates, random_ordered],
                                    index=transactions.columns).T



    combined_samples = random_neg_transactions.append(popular_neg_transactions)


    # Remove random negative samples that actually coincide with positives
    df = combined_samples[
        ~combined_samples.set_index(["customer_id", "article_id"]).index.isin(positive_pairs)
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

    # multi_encoder.create_encoder("article_id")
    multi_encoder.create_encoder("customer_id")
    #
    # transactions_processed = multi_encoder.encode("article_id", "article_id", transactions_processed)
    transactions_processed = multi_encoder.encode("customer_id", "customer_id", transactions_processed)
    transactions_processed.to_pickle("transactions_pp.pkl")
    return transactions_processed


def preprocess_articles(articles, transactions):
    print("preprocessing articles")
    articles_processed = articles[PARAM["article"]].copy()
    # articles_processed = multi_encoder.encode("article_id", "article_id", articles_processed)
    articles_processed = articles_processed.fillna(0)

    # encoding index_code
    multi_encoder.create_encoder("index_code")
    articles_processed = multi_encoder.encode("index_code", "index_code", articles_processed)


    # extract materials
    print("     extracting materials...")
    articles_processed = extract_article_material(articles_processed)
    multi_encoder.create_encoder("material")
    # articles_processed = multi_encoder.encode("material", "material", articles_processed)
    print("     extraction done!")

    # extract seasons
    print("     extracting seasons ...")
    articles_processed = extract_season(transactions, articles_processed)
    multi_encoder.create_encoder("season")
    # # articles_processed = multi_encoder.encode("season", "season", articles_processed)
    print("     extraction done!")



    print("     extracting price category ...")
    articles_processed = extract_price_category(transactions, articles_processed)
    multi_encoder.create_encoder("price_cat")
    # articles_processed = multi_encoder.encode("price_cat", "price_cat", articles_processed)
    print("     extraction done!")

    # combine material-season
    multi_encoder.create_encoder("material_season")
    articles_processed = combine_features(articles_processed, "material", "season")
    articles_processed = multi_encoder.encode("material_season", "material_season", articles_processed)
    articles_processed.drop(columns=["material"])

    # combine material-season-product_type
    # multi_encoder.create_encoder("material_season_product_type_name")
    # articles_processed = combine_features(articles_processed, "material_season", "product_type_name")
    # articles_processed = multi_encoder.encode("material_season_product_type_name", "material_season_product_type_name", articles_processed)
    # articles_processed.drop(columns=["material_season", "product_type_name"])

    # combine material-product_type
    # multi_encoder.create_encoder("material_product_type_name")
    # articles_processed = combine_features(articles_processed, "material", "product_type_name")
    # articles_processed = multi_encoder.encode("material_product_type_name", "material_product_type_name", articles_processed)
    # articles_processed.drop(columns=["material", "product_type_no"])


    # combine color-season
    multi_encoder.create_encoder("colour_group_code_season")
    articles_processed = combine_features(articles_processed, "colour_group_code", "season")
    articles_processed = multi_encoder.encode("colour_group_code_season", "colour_group_code_season", articles_processed)

    articles_processed = articles_processed.fillna(0)


    # save pp to pkl
    articles_processed.to_pickle("articles_pp.pkl")

    return articles_processed


def preprocess_customers(customers):
    print("preprocessing customers")
    customers_processed = customers[PARAM["customer"]].copy()
    # multi_encoder.create_encoder("postal_code")
    customers_processed = multi_encoder.encode("customer_id", "customer_id", customers_processed)
    # customers_processed = multi_encoder.encode("postal_code", "postal_code", customers_processed)
    customers_processed = customers_processed.fillna(20)
    customers_processed.to_pickle("customers_pp.pkl")
    return customers_processed


def preprocess_data():
    transactions, articles, customers = load_data()
    transactions = preprocess_transactions(transactions)
    articles = preprocess_articles(articles, transactions)
    customers = preprocess_customers(customers)
    # transactions, articles, customers, s = load_pkl()
    samples = create_sample(transactions, articles, customers)

    return transactions, articles, customers, samples




def recent_sales_k_months(transactions, articles, months, top_k=100):
    # Calculate age of transactions in months
    dated_transactions = transactions.loc[transactions["ordered"] == 1][["article_id", "t_dat"]]
    dated_transactions["t_dat"] = pd.to_datetime(dated_transactions["t_dat"])
    latest_date = dated_transactions["t_dat"].max()
    dated_transactions["transaction_age_months"] = 12 * (latest_date.year - dated_transactions["t_dat"].dt.year) + (
                latest_date.month - dated_transactions["t_dat"].dt.month)

    # Getting all sold items within the last 6 months
    recent_items = dated_transactions.loc[dated_transactions["transaction_age_months"] < months]
    recent_items = recent_items["article_id"].value_counts().rename_axis("article_id").reset_index(name="count")
    # recent_items = recent_items.sort_values(by="count", ascending=False)["article_id"].head(100)
    recent_items = recent_items.sort_values(by="count", ascending=False).head(top_k)
    recent_items.drop(columns=["count"], inplace=True)
    recent_items = recent_items.merge(articles, how="inner", on="article_id")

    return recent_items

def popular_articles(transactions, articles, top_k=100):
    popular_items = transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="count").sort_values(by="count", ascending=False).head(top_k)
    total_sum = popular_items["count"].sum()
    popular_items["s_weight"] = popular_items["count"] / total_sum
    prices = transactions[["article_id", "price"]].merge(popular_items[["article_id"]], on=["article_id"], how='inner').groupby(["article_id"]).min("price")
    popular_items = popular_items.merge(articles, on=["article_id"], how="inner")
    popular_items = popular_items.merge(prices, on=["article_id"], how='inner')
    return popular_items.drop(columns=["count"])

def most_active_customers(transactions, top_k=1000):
    active_customers = transactions["customer_id"].value_counts().rename_axis("customer_id").reset_index(name="count").sort_values(by="count", ascending=False).head(top_k)
    total_sum = active_customers["count"].sum()
    active_customers["s_weight"] = active_customers["count"] / total_sum
    return active_customers.drop(columns=["count"])


def repurchase_items(transactions, customers, top_k=10):
    repurchased = pd.DataFrame(columns=["customer_id","article_id"])
    for index, customer in customers.iterrows():
        purchased_items = transactions.loc[transactions["customer_id"] == customer["customer_id"]][["customer_id", "article_id"]]
        counts = purchased_items["article_id"].value_counts().rename_axis("article_id").reset_index(name="count").sort_values(by="count", ascending=False).head(top_k).drop(columns=["count"])
        counts["customer_id"] = customer["customer_id"].astype(int)
        repurchased = repurchased.append(counts)

    return repurchased


def candidates_selection(transactions, articles, customers, samples, candidates=None):
    print("selecting candidates ...")
    recent_items = recent_sales_k_months(samples, articles, 2)
    recent_items_samples = pd.merge(transactions, recent_items["article_id"], on=["article_id"], how="inner")
    # purchased_items = repurchase_items(transactions, customers)
    candidates = {
        "customers": customers["customer_id"].values.tolist(),
        "sims": []
    }
    # for index, customer in enumerate(candidates["customers"]):
    #     candidates["sims"].append(recent_items["article_id"].values.tolist())
        # candidates["sims"][index].append(purchased_items.loc[purchased_items["customer_id"] == customer]["article_id"].values.tolist())
    samples = pd.concat([samples, recent_items_samples])
    # candidates = k_nn_clustering(transactions, articles, customers, candidates)
    return samples


def k_nn_clustering(transactions, articles, customers, candidates=None, start=0):
    # prepare transaction data
    sales = transactions.merge(articles, how="inner", on='article_id')
    sales.drop(columns=["sales_channel_id", "t_dat"], inplace=True)
    articles = articles.merge(transactions[["article_id", "price"]].copy(), how="inner", on="article_id")
    # prepare candidates dict
    if candidates is None:
        candidates = {"customers": [], "sims": []}
    c = list(customers["customer_id"].values)
    for customer in range(start, len(c)):
        if customer % 50 == 0:
            if customer != start:
                print(f"{customer}/{len(c)}")
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
        candidates["sims"].append(sim_df["article_id"].head(50).values.tolist())
    return candidates


def rank_candidates(articles, samples):
    samples = samples[PARAM["samples"]].copy()
    samples["customer_id"] = pd.to_numeric(samples["customer_id"])
    samples["article_id"] = pd.to_numeric(samples["article_id"])
    samples["ordered"] = pd.to_numeric(samples["ordered"])


    # setting up data
    validation_data, training_data = split_samples(samples, 1)

    training_data = training_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)

    qids_train = training_data.groupby(["transaction_age_weeks", "customer_id"])["article_id"].count().values
    qids_validation = validation_data.groupby(["transaction_age_weeks", "customer_id"])["article_id"].count().values

    training_data = training_data.drop(["t_dat", "transaction_age_weeks"], axis=1)
    x_train = training_data.drop(columns=["ordered", "customer_id"])
    y_train = training_data["ordered"]

    validation_data = validation_data.drop(["t_dat", "transaction_age_weeks"], axis=1)
    x_test = validation_data.drop(columns=["ordered", "customer_id"])
    y_test = validation_data["ordered"]

    ranker = lgb.LGBMRanker(
        objective=FIXED_PARAMS["objective"],
        metric=FIXED_PARAMS["metric"],
        boosting_type=FIXED_PARAMS["boosting"],
        importance_type=FIXED_PARAMS["importance_type"],
        n_jobs=FIXED_PARAMS["n_jobs"],
        n_estimators=SEARCH_PARAMS["estimators"],
        verbose=SEARCH_PARAMS["verbose"],
        min_child_samples=SEARCH_PARAMS["child"],
        max_depth=SEARCH_PARAMS["depth"],
        learning_rate=SEARCH_PARAMS["learning_rate"],
        subsample=SEARCH_PARAMS["subsample"],
        num_leaves=SEARCH_PARAMS["leaves"],

    )
    print("Training ranker ...")
    if PARAM["eval"]:
        ranker.fit(
            X=x_train,
            y=y_train,
            group=qids_train,
            eval_set=[(x_train, y_train),(x_test, y_test)],
            eval_group=[qids_train,qids_validation],
            eval_names=["training set","validation set"],
            eval_at=12,

        )
        lgb.plot_metric(ranker)
        plt.show()
    else:
        ranker.fit(
            X=x_train,
            y=y_train,
            group=qids_train,
        )



    for i in ranker.feature_importances_.argsort()[::-1]:
        print(ranker.feature_name_[i], ranker.feature_importances_[i] / ranker.feature_importances_.sum())


    validation_data["prediction"] = ranker.predict(x_test)
    predictions_for_customer = validation_data.sort_values(['customer_id', 'prediction'], ascending=False).groupby('customer_id')['article_id'].apply(list).to_dict()

    print("Ranking candidates")
    # candidates["prediction"] = []
    popular_12 = list(recent_sales_k_months(transactions, articles, 2, 12)["article_id"])

    submission = pd.read_csv('../../sample_submission.csv')
    submission = multi_encoder.encode("customer_id", "customer_id", submission)
    predictions = []
    for customer_id in submission["customer_id"]:
        prediction = predictions_for_customer.get(customer_id, [])
        prediction = prediction + popular_12
        predictions.append(prediction[:12])

    predictions = [' '.join(['0' + str(p) for p in ps]) for ps in predictions]
    submission["prediction"] = predictions
    submission = multi_encoder.decode_df("customer_id", "customer_id", submission)
    submission.to_csv(f'submission.csv.gz', index=False)

    return submission


def save_candidates(candidates):
    df = pd.DataFrame(candidates)
    df.to_csv('candidates1.csv', mode='a', index=False, header=False)


def load_candidates(file="candidates1.csv"):
    candidates = pd.read_csv(file)
    candidates["sims"] = candidates["sims"].apply(literal_eval)
    candidates = candidates.to_dict(orient='list')
    return candidates


def create_submission(candidates, first=True):
    # create dataframe
    submission = pd.DataFrame(columns=["customer_id", "predictions"])
    submission["customer_id"] = candidates["customers"]
    # submission["prediction"] = candidates["prediction"]

    # label decode
    submission = multi_encoder.decode_df("customer_id", "customer_id", submission)
    submission = submission.reset_index()
    for prediction in range(len(candidates["predictions"])):
        dataframe = pd.DataFrame(columns=["article_id"])
        dataframe["article_id"] = candidates["predictions"][prediction]
        dataframe = multi_encoder.decode_df("article_id", "article_id", dataframe)
        submission.at[prediction, 'predictions'] = list(dataframe["article_id"])

    submission.drop(columns=["index"], inplace=True)
    if first:
        submission.to_csv('submission.csv', index=False, header=True, mode='w')
    else:
        submission.to_csv('submission.csv', index=False, header=False, mode="a")



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

    # candidate ranking
    candidates = rank_candidates(articles, candidates)

    # map@12 score
    if PARAM["score"]:
        map_at_12(samples, candidates)


    # # write submission
    # if PARAM["SUBMIT"]:
    #     create_submission(candidates)

