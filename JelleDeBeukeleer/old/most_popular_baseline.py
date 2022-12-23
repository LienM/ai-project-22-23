import numpy as np
import pandas as pd

transactions_file = "../data/transactions_train.csv"
customers_file = "../data/customers.csv"

# transactions_df = pd.read_csv(transactions_file)
customers_df = pd.read_csv(customers_file)

# transactions_df = transactions_df['article_id']
customers_df = customers_df['customer_id'].unique()


def get_most_popular_items(df: pd.DataFrame, recommendation_count: int = 12):
    purchase_counts = df.value_counts()
    purchase_counts.sort_values(inplace=True, ascending=False)
    return [str(aid) for aid in purchase_counts[:recommendation_count].index]


def write_most_popular(customer_ids: pd.Series, pop_items: list,
                       of_name: str = "predictions.csv"):
    of = open(of_name, 'w')
    first_line = "customer_id,prediction\n"
    of.write(first_line)
    pop_line = ""
    for item in pop_items:
        pop_line += "0" + item + " "
    pop_line = pop_line[:-1]
    for cust_id in customer_ids:
        line = str(cust_id) + "," + pop_line + "\n"
        of.write(line)
    of.close()


# print(get_most_popular_items(transactions_df))
most_popular = ['706016001', '706016002', '372860001', '610776002', '759871002', '464297007', '372860002', '610776001', '399223001', '706016003', '720125001', '156231001']
write_most_popular(customers_df, most_popular)
