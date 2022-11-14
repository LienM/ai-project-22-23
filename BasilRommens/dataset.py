import joblib
import pandas as pd
from sklearn import preprocessing

from BasilRommens.cleaning import clean_articles, clean_customers, \
    clean_transactions


def create_samples(articles, customers, transactions, samples):
    # Adapted from: https://www.kaggle.com/code/paweljankiewicz/hm-create-dataset-samples
    # This extracts three sampled datasets, containing 0.1%, 1% and 5% of all users and their transactions, and the associated articles.
    for sample_repr, sample in samples:
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
            f"../data/customers_sample{sample_repr}.feather")
        transactions_sample.to_feather(
            f"../data/transactions_sample{sample_repr}.feather")
        articles_sample.to_feather(
            f"../data/articles_sample{sample_repr}.feather")


def read_data_set(read_type):
    if read_type == 'feather':
        articles = pd.read_feather('../data/articles.feather')
        customers = pd.read_feather('../data/customers.feather')
        transactions = pd.read_feather('../data/transactions.feather')
    elif read_type == 'csv':
        articles = pd.read_csv('../data/articles.csv')
        customers = pd.read_csv('../data/customers.csv')
        transactions = pd.read_csv('../data/transactions.csv')
    return articles, customers, transactions


def prepare_feather_datasets():
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

    create_samples(articles, customers, transactions,
                   [("01", 0.001), ("1", 0.01), ("5", 0.05)])

    # store full feather files
    articles.to_feather('../data/articles.feather')
    customers.to_feather('../data/customers.feather')
    transactions.to_feather('../data/transactions.feather')

    joblib.dump(customer_encoder, '../data/customer_encoder.joblib')


def part_data_set(size):
    articles = pd.read_feather(f'../data/articles_sample{size}.feather')
    customers = pd.read_feather(f'../data/customers_sample{size}.feather')
    transactions = pd.read_feather(f'../data/transactions_sample{size}.feather')
    return articles, customers, transactions
