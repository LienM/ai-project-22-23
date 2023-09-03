import numpy as np
import pandas as pd
import seaborn as sns
from Parameters import PARAM
from Pipeline import preprocess_data, load_pkl
import matplotlib.pyplot as plt
from utils import *


def eda_materials(articles):
    material_count = articles["material"].value_counts().rename_axis("material").reset_index(name="count")
    material_count = material_count.sort_values(by="count", ascending=False).head(10)

    sns.barplot(material_count, x="material", y='count')
    plt.show()
    print(articles["material"].describe())
    print(material_count)


def eda_repurchase(transactions):
    repurchase_count = transactions.groupby(["customer_id"])["article_id"].value_counts()
    print(repurchase_count)


def eda_colour_season(transactions, articles):
    articles = extract_season(transactions, articles)
    transactions = transactions.merge(articles, on=["article_id"], how="inner")
    transactions["season"].value_counts().plot.bar()
    plt.show()
    transactions["colour_group_name"].value_counts().plot.bar()
    plt.show()

    transactions = combine_features(transactions, "season", "colour_group_name")
    transactions = transactions["season_colour_group_name"].value_counts().rename_axis("season_colour_group_name").reset_index(name="count")
    transactions = transactions.sort_values(by="count", ascending=False).head(6)
    sns.barplot(transactions, x="season_colour_group_name", y="count")

    plt.show()

def eda_material_product_type(transactions, articles):
    print(articles["product_type_name"].describe())
    print(articles["material_product_type_name"].describe())

    transactions = transactions.merge(articles, on=["article_id"], how="inner")
    print(transactions["material_product_type_name"].describe())

    transactions = transactions["material_product_type_name"].value_counts().rename_axis("material_product_type_name").reset_index(name="count")
    transactions = transactions.sort_values(by="count", ascending=False).head(10)

    ax = sns.barplot(transactions, x="material_product_type_name", y="count")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

    plt.show()

def eda_material_season(transactions, articles):
    print(articles["season"].describe())
    print(articles["material_season"].describe())

    transactions = transactions.merge(articles, on=["article_id"], how="inner")
    print(transactions["material_season"].describe())
    transactions = transactions["material_season"].value_counts().rename_axis(
        "material_season").reset_index(name="count")

    transactions = transactions.sort_values(by="count", ascending=False).head(10)

    ax = sns.barplot(transactions, x="material_season", y="count")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

    plt.show()


def eda_material_season_product_type_name(transactions, articles):

    transactions = transactions.merge(articles, on=["article_id"], how="inner")
    print(transactions["material_season_product_type_name"].describe())

    transactions = transactions["material_season_product_type_name"].value_counts().rename_axis(
        "material_season_product_type_name").reset_index(name="count")
    transactions = transactions.sort_values(by="count", ascending=False).head(20)

    ax = sns.barplot(transactions, x="material_season_product_type_name", y="count")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

    plt.show()


def eda_price(transactions, articles):
    prices = transactions[["article_id", "price"]].groupby(["article_id"]).min("price")
    prices["price"] = -np.log(prices["price"])


    # sns.distplot(x=prices.values)
    articles = articles["price_cat"].value_counts().rename_axis("price_cat").reset_index(name="count")
    articles = articles.sort_values(by="count", ascending=False).head(20)

    ax = sns.barplot(articles, x="price_cat", y="count")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

    print(prices.info)


def bucket_ages(customers, start=16 , stop=60, bucket_size=10):
    """
    Convert the ages to buckets to provide generalization
    """
    customers['age_cat'] = None
    current = start
    while current <= stop:
        label = f'{current}-{current+10}'
        condition = (customers['age'] >= current) & (customers['age'] < current+10)
        customers.loc[condition, 'age_cat'] = label
        current += 10
    condition = customers['age'] >= 66
    customers.loc[condition, 'age_cat'] = '66+'

    return customers


def combine_features(df, feature_1, feature_2):
    """
    Basic feature combining function
    : df: the dataframe containing the features
    : feature_1: the column name of the first feature
    : feature_2: the column name of the second feature
    """
    df[f"{feature_1}_{feature_2}"] = (df[feature_1].astype(str) + '_' + df[feature_2].astype(str))
    return df


def eda_age_postal_relation(customers, articles):
    customers['age'] = customers['age'].fillna(20)
    customers = bucket_ages(customers)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    customers['postal_code'] = encoder.fit_transform(customers['postal_code'])
    customers = combine_features(customers, 'age_cat', 'postal_code')
    customers.rename(columns={'age_cat_postal_code': 'postal_age'}, inplace=True)

    ages = customers['age_cat'].value_counts().rename_axis("age_cat").reset_index(name="count")
    ax = sns.barplot(ages, x="age_cat", y="count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    print(customers['postal_code'].info())



if __name__ == "__main__":

    articles = pd.read_csv('../data/articles_sample5.csv.gz')
    customers = pd.read_csv('../data/customers_sample5.csv.gz')
    # transactions = pd.read_csv('../data/transactions_sample5.csv.gz')

    eda_age_postal_relation(customers, articles)

    # transactions, articles, customers, samples = None, None, None, None
    # load dataset
    # if PARAM["PP"]:
    #     transactions, articles, customers, samples = preprocess_data()
    # else:
    #     transactions, articles, customers, samples = load_pkl()

    # eda_materials(articles)
    # eda_material_product_type(transactions,articles)
    # eda_material_season(transactions, articles)
    # eda_material_season_product_type_name(transactions, articles)
    # eda_repurchase(transactions)
    # eda_colour_season(transactions, articles)
    # eda_price(transactions, articles)
    # clean()