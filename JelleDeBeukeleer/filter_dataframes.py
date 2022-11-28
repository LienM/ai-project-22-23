import pandas as pd

transactions_file = "../data/transactions_train.csv"
customers_file = "../data/customers.csv"
articles_file = "../data/articles.csv"
transactions_file_pruned = "../data/transactions_train_pruned.csv"
customers_file_pruned = "../data/customers_pruned.csv"
articles_file_pruned = "../data/articles_pruned.csv"
params_file = "../data/model_params.json"


def prune_transactions():
    df = pd.read_csv(transactions_file)
    # df.drop('sales_channel_id', axis=1, inplace=True)
    df.to_csv(transactions_file_pruned, index=False)


def prune_articles():
    df = pd.read_csv(articles_file)
    df.drop(['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_no',
             'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
             'perceived_colour_value_id', 'perceived_colour_value_name',
             'perceived_colour_master_id', 'perceived_colour_master_name',
             'department_no', 'department_name', 'index_code', 'index_name',
             'index_group_no', 'index_group_name', 'section_no', 'section_name',
             'garment_group_no', 'garment_group_name', 'garment_group_no'], axis=1, inplace=True)
    df.to_csv(articles_file_pruned, index=False)


def prune_customers():
    df = pd.read_csv(customers_file)
    df.drop(['FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code'], axis=1, inplace=True)
    df.to_csv(customers_file_pruned, index=False)


prune_articles()
prune_customers()
prune_transactions()
