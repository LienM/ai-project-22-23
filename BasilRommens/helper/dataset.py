import random

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from BasilRommens.helper.cleaning import clean_articles, clean_customers, \
    clean_transactions
from BasilRommens.helper.helper import cosine_similarity
from BasilRommens.notebook.popularity import get_n_popular_articles


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


def impute_age(transactions):
    """
    impute the age of the customers where age is -1 based on highest cosine
    similarity between the product types bought by the to age impute customer
    and the age which it could belong to which is not -1
    :param transactions: transactions dataframe
    :return: transactions with imputed ages
    """
    # get the customers with age -1 and make an age map dict for them
    # this is to avoid overlap
    customers = transactions[transactions['age'] == -1]['customer_id'].unique()
    customer_age_map = {customer: -1 for customer in customers}

    # create age vectors for 'true' ages
    all_ages = transactions[transactions['age'] != -1]
    all_age_dict = all_ages \
        .groupby(['age', 'product_type_no'])['product_type_no'] \
        .count() \
        .to_dict()

    # determine the product type dim
    product_type_dim = max(transactions['product_type_no'].values) + 1

    age_vectors = {age: np.zeros(product_type_dim)
                   for age in all_ages['age'].unique()}
    # get the product type no counts for each age and store them in the age
    # vector dict
    for (age, product_type_no), count in tqdm(all_age_dict.items()):
        age_vectors[age][product_type_no] = count

    # create customer vectors for the customers with age -1
    customer_counts = transactions[transactions['age'] == -1] \
        .groupby(['customer_id', 'product_type_no'])['product_type_no'] \
        .count() \
        .to_dict()

    # create per customer a vector of product type counts
    customer_vectors = {customer: np.zeros(product_type_dim)
                        for customer in customers}
    for (customer_id, product_type_no), count in tqdm(customer_counts.items()):
        customer_vectors[customer_id][product_type_no] = count

    # get age from most similar age group for each customer
    for customer in tqdm(customers):
        # get the customer vector
        customer_vector = customer_vectors[customer]

        # calculate the similarity per age group and store in dict
        similarity = {age: cosine_similarity(customer_vector, age_vector)
                      for age, age_vector in age_vectors.items()}

        # fetch the age with the maximum similarity
        customer_age_map[customer] = max(similarity, key=similarity.get)

    # change the ages by taking only the -1 ages and merging them with the
    # non changing 'real' ages
    neg_ages = transactions[transactions['age'] == -1]  # -1 ages
    neg_ages['age'] = neg_ages['customer_id'].map(customer_age_map)

    pos_ages = transactions[transactions['age'] != -1]  # 'real' ages

    # concatenate both groups
    transactions = pd.concat([neg_ages, pos_ages])

    return transactions


def get_relevant_cols(articles, customers, transactions,
                      transaction_extra_cols=[]):
    """
    get the relevant columns for each of the dataframes to use for training the
    model
    :param articles: articles dataframe
    :param customers: customers dataframe
    :param transactions: transactions dataframe
    :param transaction_extra_cols: the extra columns to include for transactions
    :return: articles, customers, and transactions
    """
    # define all the relevant columns per dataframe
    relevant_article_cols = ['article_id', 'product_type_no',
                             'graphical_appearance_no', 'colour_group_code',
                             'perceived_colour_value_id',
                             'perceived_colour_master_id', 'department_no',
                             'index_group_no', 'section_no', 'garment_group_no']

    relevant_customer_cols = ['customer_id', 'FN', 'Active',
                              'fashion_news_frequency', 'age', 'postal_code']

    relevant_transaction_cols = ['customer_id', 'article_id', 'price',
                                 'sales_channel_id', 'week']
    relevant_transaction_cols.extend(transaction_extra_cols)

    # take the relevant columns
    articles = articles[relevant_article_cols]
    customers = customers[relevant_customer_cols]
    transactions = transactions[relevant_transaction_cols]

    return articles, customers, transactions


def construct_X_y(df, usable_cols):
    """
    construct the X and y from the dataframe, y will always be from the ordered
    or not column
    :param df: the dataframe from which we construct X and y
    :param usable_cols: the columns to use for training
    :return: X and y split
    """
    # define the X as the usable columns
    X = df[usable_cols]

    # define the y as the ordered column
    try:
        y = df[['ordered']]
    except Exception as e:
        # if not defined then error out
        y = None

    return X, y


def get_neg_transactions(transactions):
    """
    get the transactions that never happened
    based on code in second notebook
    :param transactions: the transactions dtaframe
    :return:
    """
    # get customer article pairs that happened
    positive_pairs = list(map(tuple, transactions[
        ['customer_id', 'article_id']].drop_duplicates().values))

    # Extract real values that are necessary to construct dataset
    real_dates = transactions["t_dat"].unique()
    real_customers = transactions["customer_id"].unique()
    real_channels = transactions["sales_channel_id"].unique()
    real_weeks = transactions["week"].unique()
    # fetch n most popular articles
    real_articles = get_n_popular_articles(transactions, 100)

    # get all price and article id combinations
    article_and_price = transactions[["article_id", "price"]].drop_duplicates(
        "article_id").set_index("article_id").squeeze()

    # How many negatives do we need to sample?
    num_neg_pos = transactions.shape[0]
    # Afterwards, we need to remove potential duplicates, so we'll sample too
    # many.
    num_neg_samples = int(num_neg_pos * 1.1)

    # Sampling negatives by selecting random users, articles, dates and sales
    # channel:
    # Note: This is quite naive. Some articles may not even have been available
    # at the date we are sampling.
    random.seed(42)

    # Sample each of the independent attributes.
    neg_dates = np.random.choice(real_dates, size=num_neg_samples)
    neg_articles = np.random.choice(real_articles, size=num_neg_samples)
    neg_customers = np.random.choice(real_customers, size=num_neg_samples)
    neg_channels = np.random.choice(real_channels, size=num_neg_samples)
    neg_weeks = np.random.choice(real_weeks, size=num_neg_samples)
    ordered = np.array([0] * num_neg_samples)
    # Assign to every article a real price.
    neg_prices = article_and_price[neg_articles].values

    # create a dataframe of negative transactions
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

    # return the negative transactions
    return chosen_neg_transactions
