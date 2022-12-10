# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import pandas as pd
from paths import path


def customer_id_to_int(x): return int(x[-16:], 16)


def simplify_transactions(transactions):
    """
    Simplify the transaction dataset, this is the most important one.
    We convert the article and customer ids to integers as weel as price to 32 bits and sales channel id to 8 bits.
    From the datetime I only week the week number
    :param transactions: the transactions
    :return: None
    """
    transactions.info(memory_usage='deep')

    transactions['customer_id'] = transactions['customer_id'].apply(customer_id_to_int).astype('int32')
    transactions['article_id'] = transactions['article_id'].astype('int32')

    # By default, I use only week, but you can add days, years and such with the following lines
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d')
    # transactions['year'] = (transactions['t_dat'].dt.year - 2000).astype('int8')
    # transactions['month'] = transactions['t_dat'].dt.month.astype('int8')
    # transactions['day'] = transactions['t_dat'].dt.day.astype('int8')

    # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
    transactions['week'] = (104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7).astype('int8')
    transactions.drop('t_dat', axis=1, inplace=True)

    transactions['price'] = transactions['price'].astype('float32')
    transactions['sales_channel_id'] = transactions['sales_channel_id'].astype('int8')

    transactions.info(memory_usage='deep')


def simplify_customers(customers):
    """
    Simplify the customer dataset, we just convert stuff to smaller integer to save those precious bits
    As described below, this does some additional preprocessing like filling in missing values with mean
    :param customers: the customers
    :return: None
    """
    customers.info(memory_usage='deep')

    # FN / Active are mapped to i8 with values 0 and 1
    customers.fillna({"FN": 0, "Active": 0}, inplace=True)

    # Everything that has weird indices, is factorized, similar to label encoding
    customers["FN"] = customers["FN"].astype('int8')
    customers["Active"] = customers["Active"].astype('int8')
    customers["club_member_status"] = pd.factorize(customers["club_member_status"])[0].astype('int8')
    customers["fashion_news_frequency"] = pd.factorize(customers["fashion_news_frequency"])[0].astype('int8')
    customers['customer_id'] = customers['customer_id'].apply(customer_id_to_int).astype('int32')
    customers['postal_code'] = pd.factorize(customers['postal_code'])[0].astype('int32')

    # Take the mean of the customer's age, there aren't many so this doesn't really matter
    customers['age'].fillna(customers['age'].mean(), inplace=True)
    customers['age'] = customers["age"].astype('int8')

    customers.info(memory_usage='deep')


def simplify_articles(articles):
    """
    Simplify the article dataset, again reducing the size of the integers
    and removing some text columns that are identical to other id columns.
    This is a lossy operation, but I've found that the these textual columns have nothing to offer, even when embedded.
    :param articles: the articles
    :return: None
    """
    articles.info(memory_usage='deep')

    # Everything that has weird indices, is factorized, similar to label encoding
    articles['article_id'] = articles['article_id'].astype('int32')
    articles['graphical_appearance_no'] = pd.factorize(articles['graphical_appearance_no'])[0].astype('int8')
    articles['colour_group_code'] = articles['colour_group_code'].astype('int8')
    articles['perceived_colour_value_id'] = articles['perceived_colour_value_id'].astype('int8')
    articles['perceived_colour_master_id'] = articles['perceived_colour_master_id'].astype('int8')
    articles['department_no'] = articles['department_no'].astype('int16')
    articles['index_code'] = pd.factorize(articles['department_no'])[0].astype('int8')
    articles['index_group_no'] = articles['index_group_no'].astype('int8')
    articles['section_no'] = articles['section_no'].astype('int8')
    articles['garment_group_no'] = articles['garment_group_no'].astype('int16')

    # This only retains ['prod_name', 'detail_desc', 'product_type_name'] as string values.
    # This is what I've found to be the only useful string columns to embed but your mileage may vary.
    articles.drop(["graphical_appearance_name", "perceived_colour_value_name", "perceived_colour_master_name",
                   "index_name", "index_group_name", "section_name", "garment_group_name"], axis=1, inplace=True)

    articles.info(memory_usage='deep')


def simplify_full():
    """
    Simplify all the things! Or just some things if you want to do it in parts.
    """
    datasets = ["transactions", "customers", "articles"]
    # datasets = ["articles"]

    for dataset in datasets:
        data = pd.read_csv(path(dataset, 'original'))
        globals()[f"simplify_{dataset}"](data)
        data.to_feather(path(dataset, 'full'))


def simplify_sample(frac=0.0001):
    """
    Create a sample of the data, this is useful for testing and debugging in other parts of the code
    """
    datasets = ["transactions", "customers", "articles"]
    # datasets = ["articles"]

    for dataset in datasets:
        data = pd.read_feather(path(dataset, 'full'))
        data.sample(frac=frac).reset_index(drop=True).to_feather(path(dataset, 'sample'))


if __name__ == '__main__':
    simplify_full()
    simplify_sample()
