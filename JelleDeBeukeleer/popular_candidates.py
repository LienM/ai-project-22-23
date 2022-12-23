import pandas as pd
import random
import json

settings_file = "./settings.json"
settings = json.load(open(settings_file))
random.seed(42)

data_dir = settings["data_directory"]
candidate_dir = settings["candidate_directory"]
processed_filenames = settings["data_filenames"]["processed"]
transactions = pd.read_csv(data_dir + processed_filenames["transactions"])
prediction_week = transactions.t_dat.max() + 1


"""
Popular candidate generation is relatively simple, going by article_purchase_count for each article
and keeping some amount of top scorers
"""
print('generating global candidates')
most_popular_count = settings["popular_candidates"]
popular_items = transactions[transactions['ordered'] == 1].drop_duplicates(subset='article_id')
popular_items.sort_values(by='article_purchase_count', ascending=False, inplace=True)
popular_items.drop(
    ['customer_id', 'ordered', 'customer_purchase_count', 'average_customer_budget', 'price_discrepancy', 'age'],
    axis=1, inplace=True)
popular_items = popular_items.iloc[:most_popular_count]
popular_items.head()
counter = 0

for customers in pd.read_csv(data_dir + processed_filenames["customers"], chunksize=30000):
    customer_temp = customers[
        ['customer_id', 'average_customer_budget', 'customer_purchase_count', 'age']]
    candidate_filenames = settings["candidate_filenames"]
    customer_temp = customer_temp.merge(popular_items, how='cross')
    customer_temp = customer_temp.reindex(columns=transactions.columns)
    customer_temp["price_discrepancy"] = customer_temp["average_article_price"] - customer_temp["average_customer_budget"]
    customer_temp.t_dat = prediction_week
    customer_temp["sales_channel_id"] = 2
    customer_temp.to_csv(candidate_dir + str(counter) + candidate_filenames["popular"], index=False)
    candidate_columns = transactions.columns
    del customer_temp
    counter += 1