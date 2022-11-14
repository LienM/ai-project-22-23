import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

BASE = 'data'


def article_features(name_components=8, description_components=8):
    # Do you really need a comment to know what this stuff is doing?
    start = time.time()
    print("starting feature engineering on articles")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    articles = pd.read_feather(f"{BASE}/articles.feather")

    # Embed the product name into 16 columns
    features = model.encode(articles['prod_name'].values.tolist()).tolist()
    transformed = PCA(n_components=name_components).fit_transform(features).tolist()
    articles[[f'prod_name_{i}' for i in range(name_components)]] = pd.DataFrame(transformed)

    # Embed the detail description into 16 columns
    # Of course this column has floats, very cool!
    features = model.encode(articles['detail_desc'].map(str).values.tolist()).tolist()
    transformed = PCA(n_components=description_components).fit_transform(features).tolist()
    articles[[f'detail_desc_{i}' for i in range(description_components)]] = pd.DataFrame(transformed)

    # Merge the two columns into one before embedding which is probably better
    # aggregated = articles.apply(lambda x: f"{x['department_name']} {x['colour_group_name']}", axis=1)
    # features = model.encode(aggregated.values.tolist()).tolist()
    # transformed = PCA(n_components=8).fit_transform(features).tolist()
    # articles[[f'dep_colour_{i}' for i in range(8)]] = pd.DataFrame(transformed)

    # Drop all the columns which we just embedded
    articles.drop(['prod_name', 'detail_desc', 'product_type_name'], axis=1, inplace=True)

    articles.info(memory_usage='deep')
    articles.to_feather(f'{BASE}/articles.feather')
    print("done simplifying customers:", time.time() - start)


def transactions_features():
    start = time.time()
    print("starting feature engineering on transactions")

    transactions = pd.read_feather(f"{BASE}/transactions.feather")

    ranking = transactions \
        .groupby("week")["article_id"].value_counts() \
        .groupby("week").rank(method="dense", ascending=False) \
        .groupby("week").rename("rank").astype("int32")


