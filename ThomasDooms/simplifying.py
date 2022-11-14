# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import time
import pandas as pd

# =================================================================================================
# Be careful!
# Using these techniques is really handy, but you need to convert article_id and customer_id back
# The code can be found here
# https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/308635
# ==================================================================================================

BASE = 'data'


def customer_id_to_int(x): return int(x[-16:], 16)


def simplify_transactions():
    print("starting transaction simplifying")
    start = time.time()

    transactions = pd.read_csv(f'{BASE}/transactions.csv')
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
    transactions.to_feather(f'{BASE}/transactions.feather')
    print(f"done simplifying transactions in {time.time() - start:.2f} seconds\n\n")


def simplify_customers():
    print("starting customer simplifying")
    start = time.time()

    customers = pd.read_csv(f'{BASE}/customers.csv')
    customers.info(memory_usage='deep')

    # FN / Active are mapped to i8 with values 0 and 1
    customers.fillna({"FN": 0, "Active": 0}, inplace=True)

    # Everything that has weird indices, is factorized, similar to label encoding
    customers["FN"] = customers["FN"].astype('int8')
    customers["Active"] = customers["Active"].astype('int8')
    customers["club_member_status"] = pd.factorize(customers["club_member_status"])[0].astype('int8')
    customers["fashion_news_frequency"] = pd.factorize(customers["fashion_news_frequency"])[0].astype('int8')
    customers['customer_id'] = customers['customer_id'].apply(customer_id_to_int).astype('int64')
    customers['postal_code'] = pd.factorize(customers['postal_code'])[0].astype('int32')

    customers.info(memory_usage='deep')
    customers.to_feather(f'{BASE}/customers.feather')
    print(f"done simplifying customers in {time.time() - start:.2f} seconds\n\n")


def simplify_articles():
    print("starting article simplifying")
    start = time.time()

    articles = pd.read_csv(f'{BASE}/articles.csv')
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
    articles.drop(["graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
                   "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name",
                   "garment_group_name"], axis=1, inplace=True)

    articles.info(memory_usage='deep')
    articles.to_feather(f'{BASE}/articles.feather')
    print(f"done simplifying articles in {time.time() - start:.2f} seconds\n\n")


def simplify_submission():
    submission = pd.read_csv(f'{BASE}/submission.csv')
    submission.to_feather(f'{BASE}/submission.feather')


if __name__ == '__main__':
    simplify_articles()
    # simplify_customers()
    # simplify_transactions()
    # simplify_submission()
