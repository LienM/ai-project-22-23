# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import os
import pickle

import pandas as pd
from lightgbm import LGBMRanker


# Choo Choo!
# def train(): pass


def train_model(train, columns):
    groups = train.groupby(["week", "customer_id"])["article_id"].count().values

    # seed = 42069  # picked by pure random chance

    # params = {
    #     "objective": "binary",
    #     "boosting": "gbdt",
    #     "max_depth": -1,
    #     "num_leaves": 40,
    #     "subsample": 0.8,
    #     "subsample_freq": 1,
    #     "bagging_seed": seed,
    #     "learning_rate": 0.05,
    #     "feature_fraction": 0.6,
    #     "min_data_in_leaf": 100,
    #     "lambda_l1": 0,
    #     "lambda_l2": 0,
    #     "random_state": seed,
    #     "metric": "auc",
    #     "verbose": -1
    # }

    print(train)

    # Actually train the model
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        n_estimators=20,
        importance_type="gain",
        force_row_wise=True,
        verbose=10
    )

    model = model.fit(
        train[columns],  # X
        train["purchased"],  # y
        group=groups,
    )

    # Show the impact of each feature on the model
    for i in model.feature_importances_.argsort()[::-1]:
        print(columns[i], model.feature_importances_[i] / model.feature_importances_.sum())
    return model
