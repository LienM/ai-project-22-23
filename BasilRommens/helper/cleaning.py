import numpy as np
import pandas as pd
from scipy.stats import rankdata


def clean_articles(articles):
    """
    cleans all the articles to use up less memory
    :param articles: the articles dataframe to clean
    :return: cleaned articles dataframe
    """
    # mainly change type ot integers to save memory
    articles['article_id'] = articles['article_id'].astype(np.uint32)
    articles['product_code'] = articles['product_code'].astype(np.int32)
    articles['prod_name'] = articles['prod_name'].astype(str)
    articles['product_type_no'] = articles['product_type_no'].astype(np.int16)
    articles['product_type_name'] = articles['product_type_name'].astype(str)
    articles['product_group_name'] = articles['product_group_name'].astype(str)
    articles['graphical_appearance_no'] = articles[
        'graphical_appearance_no'].astype(np.int32)
    articles['graphical_appearance_name'] = articles[
        'graphical_appearance_name'].astype(str)
    articles['colour_group_code'] = articles['colour_group_code'].astype(
        np.int8)
    articles['colour_group_name'] = articles['colour_group_name'].astype(str)
    articles['perceived_colour_value_id'] = articles[
        'perceived_colour_value_id'].astype(np.int8)
    articles['perceived_colour_value_name'] = articles[
        'perceived_colour_value_name'].astype(str)
    articles['perceived_colour_master_id'] = articles[
        'perceived_colour_master_id'].astype(np.int8)
    articles['perceived_colour_master_name'] = articles[
        'perceived_colour_master_name'].astype(str)
    articles['department_no'] = articles['department_no'].astype(np.uint16)
    articles['department_name'] = articles['department_name'].astype(str)
    articles['index_code'] = articles['index_code'].astype(str)
    articles['index_name'] = articles['index_name'].astype(str)
    articles['index_group_no'] = articles['index_group_no'].astype(np.uint8)
    articles['index_group_name'] = articles['index_group_name'].astype(str)
    articles['section_no'] = articles['section_no'].astype(np.uint8)
    articles['section_name'] = articles['section_name'].astype(str)
    articles['garment_group_no'] = articles['garment_group_no'].astype(
        np.uint16)
    articles['garment_group_name'] = articles['garment_group_name'].astype(str)
    articles['detail_desc'] = articles['detail_desc'].astype(str)

    return articles


def clean_customers(customers):
    """
    cleans all the customers to use up less memory
    :param customers: customers dataframe to clean
    :return: cleaned customers dataframe
    """
    # mainly change type ot integers to save memory
    customers['customer_id'] = customers['customer_id'].astype(str)
    customers['FN'] = customers['FN'].replace(np.nan, 0)
    customers['FN'] = customers['FN'].astype(np.uint8)
    customers['Active'] = customers['Active'].replace(np.nan, 0)
    customers['Active'] = customers['Active'].astype(np.uint8)
    customers['club_member_status'] = customers['club_member_status'].astype(
        str)

    # convert all the none like values to 0
    customers['fashion_news_frequency'] = customers[
        'fashion_news_frequency'].replace(['NONE', np.nan, 'None'], 0)
    # convert all the monthly values to 1
    customers['fashion_news_frequency'] = customers[
        'fashion_news_frequency'].replace('Monthly', 1)
    # convert all the regularly values to 2
    customers['fashion_news_frequency'] = customers[
        'fashion_news_frequency'].replace('Regularly', 2)
    # set the type to uint8
    customers['fashion_news_frequency'] = customers[
        'fashion_news_frequency'].astype(np.uint8)

    # set age to -1 if it is not a number
    customers['age'] = customers['age'].replace(np.nan, -1)
    customers['age'] = customers['age'].astype(np.int8)
    customers['postal_code'] = customers['postal_code'].astype(str)

    return customers


def clean_transactions(transactions):
    """
    cleans all the transactions to use up less memory
    :param transactions: transactions dataframe to clean
    :return: cleaned up transactions
    """
    # mainly change type ot integers to save memory
    # load in the correct date time format
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'],
                                           format='%Y-%m-%d')
    transactions['customer_id'] = transactions['customer_id'].astype(str)
    transactions['article_id'] = transactions['article_id'].astype(np.uint32)
    transactions['price'] = transactions['price'].astype(np.float16)
    transactions['sales_channel_id'] = transactions['sales_channel_id'].astype(
        np.uint8)

    # adding the weeks as a column in transactions
    transactions['week'] = transactions['t_dat'].dt.isocalendar().year * 53 + \
                           transactions['t_dat'].dt.isocalendar().week
    transactions['week'] = rankdata(transactions['week'], 'dense')

    return transactions
