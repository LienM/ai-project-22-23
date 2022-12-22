import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing

from BasilRommens.helper.cleaning import clean_articles, clean_customers, \
    clean_transactions


def create_samples(articles, customers, transactions, sample_info):
    """
    Adapted from: https://www.kaggle.com/code/paweljankiewicz/hm-create-dataset-samples
    taken from 2nd notebook of course project AI
    This extracts three sampled datasets, containing 0.1%, 1% and 5% of all
    users and their transactions, and the associated articles.
    :param articles: the articles dataframe
    :param customers: customers dataframe
    :param transactions: transactions dataframe
    :param sample_info: list of sample suffix and sample size
    :return: nothing
    """
    for sample_repr, sample in sample_info:
        customers_sample = customers.sample(int(customers.shape[0] * sample),
                                            replace=False)
        customers_sample_ids = set(customers_sample["customer_id"])
        transactions_sample = transactions[
            transactions["customer_id"].isin(customers_sample_ids)]
        articles_sample_ids = set(transactions_sample["article_id"])
        articles_sample = articles[
            articles["article_id"].isin(articles_sample_ids)]

        # resetting index
        customers_sample = customers_sample.reset_index()
        transactions_sample = transactions_sample.reset_index()
        articles_sample = articles_sample.reset_index()

        # saving files in feather format
        customers_sample.to_feather(
            f"data/customers_sample{sample_repr}.feather")
        transactions_sample.to_feather(
            f"data/transactions_sample{sample_repr}.feather")
        articles_sample.to_feather(
            f"data/articles_sample{sample_repr}.feather")


def read_data_set(read_type):
    """
    Reads the data set from the either feather or csv files.
    :param read_type: the read type of the dataset
    :return: articles, customers, transactions dataframes
    """
    articles, customers, transactions = None, None, None
    if read_type == 'feather':
        articles = pd.read_feather('data/articles.feather')
        customers = pd.read_feather('data/customers.feather')
        transactions = pd.read_feather('data/transactions.feather')
    elif read_type == 'csv':
        articles = pd.read_csv('data/articles.csv')
        customers = pd.read_csv('data/customers.csv')
        transactions = pd.read_csv('data/transactions.csv')
    return articles, customers, transactions


def create_feather_datasets():
    """
    Cleans the datasets and saves them in feather format
    :return: nothing
    """
    articles, customers, transactions = read_data_set('csv')

    # reduce memory requirements of all dataframes
    articles = clean_articles(articles)
    customers = clean_customers(customers)
    transactions = clean_transactions(transactions)

    # encode the ids
    customer_encoder = preprocessing.LabelEncoder()
    customer_encoder.fit(customers['customer_id'])
    customers['customer_id'] = customer_encoder.transform(
        customers['customer_id'])
    transactions['customer_id'] = customer_encoder.transform(
        transactions['customer_id'])
    customers['customer_id'].astype(np.uint32)
    transactions['customer_id'].astype(np.uint32)

    # store full feather files
    articles.to_feather('data/articles.feather')
    customers.to_feather('data/customers.feather')
    transactions.to_feather('data/transactions.feather')

    # save the encoder as it is needed for customer id translation
    joblib.dump(customer_encoder, 'data/customer_encoder.joblib')


def part_data_set(size_suffix):
    """
    reads a part of the dataset using the size suffix to determine the sample
    size
    :param size_suffix: a string that determines the sample size
    :return: articles, customers, transactions sample sized dataframes
    """
    articles = pd.read_feather(f'data/articles_sample{size_suffix}.feather')
    customers = pd.read_feather(f'data/customers_sample{size_suffix}.feather')
    transactions = pd.read_feather(f'data/transactions_sample{size_suffix}.feather')
    return articles, customers, transactions


if __name__ == '__main__':
    # create the feather files
    create_feather_datasets()
    articles, customers, transactions = read_data_set('feather')
    create_samples(articles, customers, transactions,
                   [("01", 0.001), ("1", 0.01), ("5", 0.05)])
