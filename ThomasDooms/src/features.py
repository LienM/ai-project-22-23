import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from paths import path


def articles_features(articles, df, name_components=8, description_components=8):
    """
    Generate features for the articles, this will mainly add the name and description embeddings
    :param articles: the articles
    :param df: the transactions but with a shorter name to make the code more readable
    :param name_components: the number of features to embed the name by
    :param description_components: the number of features to embed the description by
    :return: the new and feature enriched articles
    """
    # Looking at some benchmarks this model has the best tradeoff between speed and accuracy
    model = SentenceTransformer('all-MiniLM-L6-v2')

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
    aggregated = articles.apply(lambda x: f"{x['department_name']} {x['colour_group_name']}", axis=1)
    features = model.encode(aggregated.values.tolist()).tolist()
    transformed = PCA(n_components=2).fit_transform(features).tolist()
    articles[[f'dep_colour_{i}' for i in range(2)]] = pd.DataFrame(transformed)

    # Drop all the columns which we just embedded
    cols = ['prod_name', 'detail_desc', 'product_type_name', 'product_group_name', 'department_name', 'colour_group_name']
    articles.drop(cols, axis=1, inplace=True)

    # Below are some attempts at the seasonal feature engineering which didn't work out

    # Calculate the amount of times an article is bought per season
    fall = df[df["season"] == 0].groupby("article_id").size().reset_index(name="fall")
    # winter = df[df["season"] == 1].groupby("article_id").size().reset_index(name="winter")
    # spring = df[df["season"] == 2].groupby("article_id").size().reset_index(name="spring")
    # summer = df[df["season"] == 3].groupby("article_id").size().reset_index(name="summer")

    articles = pd.merge(articles, fall, on="article_id", how="left")
    # articles = pd.merge(articles, winter, on="article_id", how="left")
    # articles = pd.merge(articles, spring, on="article_id", how="left")
    # articles = pd.merge(articles, summer, on="article_id", how="left")

    articles["fall"] = articles["fall"].fillna(0).astype('int32')
    articles["fall"] /= articles["fall"].max()

    # seasons = ["fall", "winter", "spring", "summer"]

    # articles[seasons] = articles[seasons].fillna(0).astype('int32')
    # articles["season_var"] = articles[seasons].max(axis=1)
    # articles["season_var"] /= articles[seasons].sum(axis=1)
    # articles["season_var"] = articles["season_var"].astype('float32')

    articles.info(memory_usage='deep')
    return articles


def transactions_features(df):
    """
    Generate features for the transactions,
    this adds a custom rank feature which is a small improvement but not much,
    it also adds a season for other calculations later
    :param df: the transactions but with a shorter name to make the code more readable
    :return:
    """
    counts = df.groupby('week')['article_id'].value_counts().rank(pct=True).reset_index(name='rank')
    counts["week"] += 1
    df = pd.merge(df, counts, on=['week', 'article_id'], how='left')
    df["rank"] = df["rank"].astype('float32')

    df['season'] = df['week'].map(lambda x: (x % 52) // 13)
    df['season'] = df['season'].astype('int8')

    df.info(memory_usage='deep')
    return df


def customers_features(customers):
    """
    This is probably the most complex feature engineering part, but I've managed to write it on a single line
    :param customers: the customers
    :return: the customers with the massively improved features
    """
    return customers


def all_features(test=False):
    """
    Generate all the features for the data, I usually comment out the stuff I need to rerun specific parts
    :param test: whether to use a subset of the data for quick testing
    :return: None
    """
    transactions = pd.read_feather(path("transactions", 'sample' if test else 'full'))
    articles = pd.read_feather(path("articles", 'full'))
    customers = pd.read_feather(path("customers", 'full'))

    transactions = transactions_features(transactions)
    articles = articles_features(articles, transactions)
    customers = customers_features(customers)

    transactions.to_feather(path("transactions", 'selected' if test else 'features'))
    articles.to_feather(path("articles", 'features'))
    customers.to_feather(path("customers", 'features'))


if __name__ == '__main__':
    # Change pandas display settings to show more columns and rows
    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = 50

    all_features()
