# -*- coding: utf-8 -*-
"""data_preparation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HEkz9SCUojBSS4srUK4PqzDlxxxuYg3c
"""

import pandas as pd
import numpy as np
import random
import datetime
import json
from sklearn import preprocessing
import lightgbm as lgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gc

settings_file = "./settings.json"
settings = json.load(open(settings_file))

file_dict = settings['data_filenames']
data_dir = settings['data_directory']
dataset_size = settings["dataset_size"]

"""
Preparing file names to be used for reading/writing
"""
articles_filename = data_dir + file_dict[dataset_size]["articles"]
customers_filename = data_dir + file_dict[dataset_size]["customers"]
transactions_filename = data_dir + file_dict[dataset_size]["transactions"]
bestseller_filename = data_dir + file_dict["processed"]["bestsellers"]	

"""
reading dataset from given files
"""
gc.collect()
articles = pd.read_csv(articles_filename)
customers = pd.read_csv(customers_filename)
transactions = pd.read_csv(transactions_filename)
print(transactions.columns)

"""
Encoding of features for later models to not need to work with strings
"""
customer_encoder = preprocessing.LabelEncoder()
article_encoder = preprocessing.LabelEncoder()
postal_encoder = preprocessing.LabelEncoder()
customers['customer_id'] = customer_encoder.fit_transform(customers['customer_id'])
customers['postal_code'] = postal_encoder.fit_transform(customers['postal_code'])
articles['article_id'] = article_encoder.fit_transform(articles['article_id'])
transactions['customer_id'] = customer_encoder.transform(transactions['customer_id'])
transactions['article_id'] = article_encoder.transform(transactions['article_id'])
transactions['ordered'] = 1

"""
Preemptively drop non-used features from customers and articles to save memory
"""
pf = settings["processed_features"]
for col in customers.columns:
    if col not in pf:
        customers.drop(col, inplace=True, axis=1)
for col in articles.columns:
    if col not in pf:
        articles.drop(col, inplace=True, axis=1)

"""
Some features are based on only recent data (e.g. how popular an item has been
recently). A recent slice is taken and the calculations are only performed on this
"""
print("taking recent slice")
recent_weeks = settings["recent_weeks"]
last_day = transactions['t_dat'].max()
last_day_date = [int(i) for i in last_day.split('-')]
last_day_date = datetime.date(last_day_date[0], last_day_date[1], last_day_date[2])
min_date = last_day_date - datetime.timedelta(weeks=recent_weeks)
min_date = min_date.strftime('%Y-%m-%d')
recent_slice = transactions[transactions['t_dat'] >= min_date].copy()

print("calculating user purchase count")
recent_purchases = recent_slice.groupby('customer_id')[['article_id']].count()
recent_purchases.rename(columns={'article_id': 'customer_purchase_count'}, inplace=True)
customers = customers.merge(recent_purchases, how='left', on='customer_id')

print("calculating user budget")
recent_customer_spendings = recent_slice.groupby('customer_id')[['price']].mean()
recent_customer_spendings.rename(columns={'price': 'average_customer_budget'}, inplace=True)
customers = customers.merge(recent_customer_spendings, how='left', on='customer_id')
customers.head()

print("calculating article purchase count")
recent_purchases = recent_slice.groupby('article_id')[['article_id']].count()
recent_purchases.rename(columns={'article_id': 'article_purchase_count'}, inplace=True)
articles = articles.merge(recent_purchases, how='left', on='article_id')

"""
Article price should not be calculated on the recent slice, as this would
result in too many NaN values
"""
print("calculating average article price")
mean_article_prices = transactions.groupby('article_id')[['price']].mean()
mean_article_prices.rename(columns={'price': 'average_article_price'}, inplace=True)
articles = articles.merge(mean_article_prices, how='left', on='article_id')
customers = customers.fillna(0)
articles = articles.fillna(0)
gc.collect()
articles.head()

"""
Keep track of which items are often the first item a customer buys, can be used as 
e.g. candidate generation
"""
print("determining popular first items")
first_purchases = transactions.sort_values(by="t_dat").groupby('customer_id').first()
first_purchase_counts = first_purchases["article_id"].value_counts()
first_purchase_df = pd.DataFrame(
    data={'article_id': first_purchase_counts.index, 'first_purchase_count': first_purchase_counts.values})
articles = articles.merge(first_purchase_df, how="left")
articles = articles.fillna(0)

"""
Limiting time-range of the dataset for future processing. Both to save memory and
put more emphasis on recent information (which is likely more relevant)
"""
print("removing oldest data from transactions")
n_weeks = settings["full_weeks"]
if n_weeks >= 0:
    last_day = transactions['t_dat'].max()
    last_day_date = [int(i) for i in last_day.split('-')]
    last_day_date = datetime.date(last_day_date[0], last_day_date[1], last_day_date[2])
    min_date = last_day_date - datetime.timedelta(weeks=n_weeks)
    min_date = min_date.strftime('%Y-%m-%d')
    transactions = transactions[transactions['t_dat'] >= min_date]
gc.collect()

"""
Customer activity per past week:
Counts the amount of times a customer bought something n weeks before the training week
"""
activity_features = []
offsets = [0]
print("calculating customer activity")
min_week = settings["activity_range"][0]
step_size = settings["activity_range"][2]
max_week = settings["activity_range"][1]
for recent_weeks in range(min_week, max_week, step_size):
    col_name = "activity_" + str(recent_weeks) + "_weeks"
    last_day = transactions['t_dat'].max()
    last_day_date = [int(i) for i in last_day.split('-')]
    last_day_date = datetime.date(last_day_date[0], last_day_date[1], last_day_date[2])
    min_date = last_day_date - datetime.timedelta(weeks=recent_weeks)
    min_date = min_date.strftime('%Y-%m-%d')
    recent_view = transactions['t_dat'] >= min_date
    customer_counts = transactions[recent_view]['customer_id'].value_counts()
    temp_df = pd.DataFrame(data={
        'customer_id': customer_counts.index,
        col_name: customer_counts.values
    })
    customers = customers.merge(temp_df, how='left')
    customers = customers.fillna(0)
    temp_offset = customers[col_name]
    customers[col_name] -= offsets[-1]
    offsets = [temp_offset]
    activity_features.append(col_name)
del offsets

"""
If activity features have been calculated, determine whether or not the customer has been active in the last week
"""
if len(activity_features) > 0:
    customers['recently_active'] = customers[activity_features[0]] > 0
    activity_features.append('recently_active')
    customers['recently_active'] = customers['recently_active'].astype("bool")

"""
Convert datetime to week counter for easier processing
"""
# https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
transactions.t_dat = pd.to_datetime(transactions.t_dat, format='%Y-%m-%d')
transactions.t_dat = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days

"""Negative samples still need improvement"""

# bestseller rank from
# https://github.com/radekosmulski/personalized_fashion_recs/blob/main/02_Basic_Model_Redux.ipynb
sale_counts = transactions \
    .groupby('t_dat')['article_id'].value_counts() \
    .groupby('t_dat').rank(method='dense', ascending=False) \
    .groupby('t_dat').head(12).rename('bestseller_rank')

article_prices = articles[["article_id", "average_article_price"]]

weeks = sale_counts.index.get_level_values(0)
aid_list = sale_counts.index.get_level_values(1)
counts = sale_counts.values
sale_counts = pd.DataFrame(data={
    "article_id":aid_list,
    "t_dat":weeks,
    "bestseller_rank": counts
})
sale_counts["t_dat"] += 1
sale_counts.to_csv(bestseller_filename, index=False)

unique_transactions = transactions.groupby(['t_dat', 'customer_id']).head(1) \
    .drop(columns=['article_id', 'price']).copy()


candidates_bestsellers = pd.merge(
    unique_transactions,
    sale_counts,
    on='t_dat',
)

candidates_bestsellers = candidates_bestsellers.merge(
    article_prices,
    how="left",
    on="article_id"
)
candidates_bestsellers["price"] = candidates_bestsellers["average_article_price"]

transactions = transactions.merge(articles, how="inner", on='article_id')
transactions = pd.concat([transactions, candidates_bestsellers])
transactions = transactions.merge(customers, how="inner", on='customer_id')
transactions["ordered"].fillna(0, inplace=True)
transactions["price_discrepancy"] = transactions["average_article_price"] - transactions["average_customer_budget"]
transactions.drop_duplicates(['customer_id', 'article_id', 't_dat'], inplace=True)
transactions = transactions.merge(
    sale_counts[['t_dat', 'article_id']],
    on=['t_dat', 'article_id'],
    how='left'
)
transactions["bestseller_rank"].fillna(999, inplace=True)

"""
Only keep the relevant features, which generally are already the only features
remaining due to filtering customers and articles previously
"""
print("slicing processed transactions")
feature_names = settings["processed_features"] + activity_features
transactions_processed = transactions[feature_names]
del transactions
transactions_processed.head()
gc.collect()
transactions_processed = transactions_processed.fillna(0)

print("writing processed transactions to csv")
customers.to_csv(data_dir + settings["data_filenames"]["processed"]["customers"], index=False)
articles.to_csv(data_dir + settings["data_filenames"]["processed"]["articles"], index=False)
transactions_processed.to_csv(data_dir + settings["data_filenames"]["processed"]["transactions"], index=False)
