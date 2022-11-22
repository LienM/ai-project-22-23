# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================
import os
from os.path import exists
import random

import pandas as pd

import candidates
import features
import simplify
from paths import path
from infer import *
from train import train_model
import time

pd.options.display.max_columns = None
pd.options.display.width = None
# pd.options.display.max_rows = None
random.seed(42)

DATA = '../data'  # path to where the data is stored
MODELS = '../models'  # path to where the models are stored
FEATURES = '../features'  # path to where the feature engineered data is stored

WEEKS = 10  # number of weeks to use for training
CV = False  # whether to use cross validation or not
TESTING = True  # whether run in test mode or not

CREATE_CANDIDATES = False
TRAIN_MODEL = True
INFER_SUBMISSION = True

# =============================================================================================
# Load data
# kind = 'sample' if TESTING else 'features'

# transactions = pd.read_feather(path('transactions', kind))
articles = pd.read_feather(path('articles', 'features'))
customers = pd.read_feather(path('customers', 'features'))


# =============================================================================================
# Generating candidates

if CREATE_CANDIDATES:
    pass  # TODO


# =============================================================================================
# Training the model

# Set the concatenation candidates and positive samples as the transactions and only use last k weeks
transactions = pd.read_feather(path('candidates', 'full'))

# Increase by 1 if not using cross-validation
# Also decrease threshold by 1 if not using cross-validation
max_week = 104
test_week = max_week + int(not CV)

# Join all the things!
transactions = pd.merge(transactions, articles, on="article_id", how="left")
transactions = pd.merge(transactions, customers, on="customer_id", how="left")

# tf does this do? This code is necessary otherwise the model will not work :(
transactions.sort_values(["week", "customer_id"], inplace=True)
transactions.reset_index(drop=True, inplace=True)

# print(transactions)

# Splitting the data into train and test
train = transactions[transactions["week"] != test_week]
test = transactions[transactions["week"] == test_week]
test = test.drop_duplicates(["customer_id", "article_id", "sales_channel_id"]).copy()

print(train["postal_code"].value_counts())

# Set all the columns we use to train and test
cols = ["article_id", "sales_channel_id", "FN", "Active", "index_code"]
cols += ["perceived_colour_master_id", "graphical_appearance_no", "colour_group_code", "perceived_colour_value_id"]
cols += ["product_type_no", "department_no", "index_group_no", "section_no"]
cols += ["garment_group_no", "club_member_status", "fashion_news_frequency", "age"]
cols += ["fall", "rank", "dep_colour_0", "dep_colour_1"]
cols += [f"prod_name_{i}" for i in range(8)]
cols += [f"detail_desc_{i}" for i in range(8)]

if TRAIN_MODEL:
    # Train the model, and save it
    print("starting model training")
    model = train_model(train, cols)
    pickle.dump(model, open(path("models", "best_last_rank3"), "wb"))

# =============================================================================================
# Inferring the submission

if INFER_SUBMISSION:
    baseline = compute_baseline(transactions, test_week)
    model = pickle.load(open(path("models", "best_last_rank3"), "rb"))
    submission = pd.read_feather(path("submission", "full"))

    start = time.time()
    print(f"starting inferring submission")

    infer(model, test, submission, baseline, cols, "../submission.csv.gz", CV)
    print(f"done inferring submission in {time.time() - start:.2f} seconds\n\n")
