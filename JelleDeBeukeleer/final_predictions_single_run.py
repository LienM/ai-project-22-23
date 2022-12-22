# -*- coding: utf-8 -*-
"""generate_predictions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CV0BmCSggARp2ZdvmNwcTXlF8OY9pG-K
"""
import pandas as pd
import numpy as np
import random
import datetime
import json
import os
from sklearn import preprocessing
import lightgbm as lgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gc

from ensemble_model import CustomEnsemble, make_model

settings_file = "./settings.json"
settings = json.load(open(settings_file))

print("reading settings")
file_dict = settings['data_filenames']
data_dir = settings['data_directory']
candidate_dir = settings["candidate_directory"]
prediction_dir = settings["predictions_directory"]
candidate_files = [str(f) for f in os.listdir(candidate_dir)
                   if os.path.isfile(os.path.join(candidate_dir, f))]

recommendation_count = settings["recommendation_count"]
dataset_size = settings["dataset_size"]
dataset_type = "processed"

transactions_filename = data_dir + file_dict[dataset_type]["transactions"]

articles_filename = data_dir + file_dict[dataset_size]["articles"]
customers_filename = data_dir + file_dict[dataset_size]["customers"]
print("reading dataframes")
articles = pd.read_csv(articles_filename)
customers = pd.read_csv(customers_filename)
customer_encoder = preprocessing.LabelEncoder()
article_encoder = preprocessing.LabelEncoder()
customer_encoder.fit(customers['customer_id'])
article_encoder.fit(articles['article_id'])
del articles
del customers
gc.collect()

transactions = pd.read_csv(transactions_filename)
transactions["ordered"] = transactions["ordered"].astype("int64")
training_weeks = settings["training_weeks"]
max_week = transactions["t_dat"].max() + 1
transactions = transactions[(max_week - transactions["t_dat"]) <= training_weeks]
transactions.sort_values("customer_id", inplace=True)

print("training model")
y = transactions["ordered"]
x = transactions.drop(['ordered', "customer_id"], axis=1)
df_columns = x.columns
model = make_model(x)

model = model.fit(x, y)
del transactions

customer_chunks = pd.read_csv(customers_filename, chunksize=300000)
customer_counter = 0

for customers in customer_chunks:
    counter = 0
    predictions_files = []
    customer_list = list(customers["customer_id"])
    customer_list = customer_encoder.transform(customer_list)
    full_df = pd.DataFrame(columns=df_columns)

    for filename in candidate_files:
        print(f"chunk: {customer_counter} predicting for", filename, end="\r")
        candidate_file = candidate_dir + filename
        candidate_items = pd.read_csv(candidate_file)
        candidate_items = candidate_items[candidate_items["customer_id"].isin(customer_list)]
        if len(candidate_items) == 0:
            continue
        predictions_file = prediction_dir + "predictions" + str(counter) + ".csv"
        counter += 1
        predictions_files.append(predictions_file)
        if "ordered" in candidate_items.columns:
            candidate_items.drop("ordered", inplace=True, axis=1)
        candidate_items[["score0", "score"]] = model.predict_proba(candidate_items.drop("customer_id", axis=1))
        candidate_items.drop("score0", inplace=True, axis=1)
        candidate_items = candidate_items[["customer_id", "article_id", "score"]]
        candidate_items.sort_values("score", inplace=True, ascending=False)
        candidate_items = candidate_items.groupby("customer_id").head(recommendation_count)

        full_df = pd.concat([full_df, candidate_items])
        del candidate_items
        gc.collect()

    full_df.sort_values("score", inplace=True, ascending=False)
    full_df.drop_duplicates(subset=["customer_id", "article_id"], inplace=True)
    full_df = full_df.groupby("customer_id").head(recommendation_count)

    full_df["customer_id"].value_counts()

    full_df = full_df[["customer_id", "article_id"]]

    full_df["customer_id"] = full_df["customer_id"].apply(int)
    full_df["article_id"] = full_df["article_id"].apply(int)
    full_df["article_id"] = article_encoder.inverse_transform(full_df["article_id"])
    full_df["customer_id"] = customer_encoder.inverse_transform(full_df["customer_id"])

    output_file = data_dir + str(customer_counter) + settings["prediction_filenames"]["final"]

    full_df['prediction'] = full_df['article_id'].apply(str)
    full_df["prediction"] = "0" + full_df["prediction"]
    full_df.groupby('customer_id').agg({'prediction': " ".join}).to_csv(output_file)
    customer_counter += 1
