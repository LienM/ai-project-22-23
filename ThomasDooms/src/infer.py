# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import pickle
import time

import pandas as pd
from itertools import accumulate


def ap_k(actual, predicted):
    """
    I'm sorry but, I wanted to write the shortest possible function to compute the average precision at k
    instead of the easiest to understand. I mean it's still readable but yeah...
    !! This function assumes actual and predicted to be of length 12 for this competition !!
    :param actual: the ground truth values
    :param predicted: the predicted values
    :return: the ap_k score
    """
    cor = list(accumulate([int(r == t) for r, t in zip(predicted, actual)]))
    return sum((r == t) * c / (i + 1) for i, (r, t, c) in enumerate(zip(predicted, actual, cor))) / min(len(actual), 12)


def map_k(actual, predicted):
    """
    Compute the map@k score, uses the ap_k function, not much else to say
    !! This function assumes actual and predicted to be of length 12 for this competition !!
    :param actual: the ground truth values
    :param predicted: the predicted values
    :return:
    """
    return sum(ap_k(a, p) for a, p in zip(actual, predicted)) / len(actual)


def compute_baseline(transactions, test_week):
    """
    Compute the baseline, this is based on popularity, so the most popular articles are predicted
    It's a single one-liner, so I don't think I need to explain it
    :param transactions: the transactions
    :param test_week: the test week
    :return: 12 most popular articles to be used as baseline
    """
    return transactions[transactions["week"] == test_week - 1]["article_id"].value_counts().head(12).index.tolist()


def infer(model, test, submission, baseline, columns, path, cv):
    """
    Infer the model, this is the main function of this file
    :param model: the lightgbm model to use
    :param test: the test data
    :param submission: the example submission data used to get the original customer ids
    :param baseline: the baseline to use
    :param columns: the columns to use as features
    :param path: the path to store the submission file
    :param cv: whether to use cross validation
    :return:
    """
    # Do the actual prediction, sort by highest to lowest per customer and then groupby customer
    # If we didn't do the sorting beforehand, the order would be wrong
    test["prediction"] = model.predict(test[columns])
    test.sort_values(["customer_id", "prediction"], ascending=False, inplace=True)
    predicted = test.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Compute the mapped customer id and join the baseline with the predictions
    submission["mapped_customer_id"] = submission["customer_id"].map(lambda x: int(x[-16:], 16)).astype("int32")
    submission["prediction"] = submission["mapped_customer_id"].apply(lambda x: (predicted.get(x, []) + baseline)[:12])

    # If we are doing cross validation, compute the map@k score, otherwise save the submission
    if cv:
        per_customer = test[test["purchased"] == 1].groupby("customer_id")["article_id"].apply(list)
        combined = pd.merge(submission, per_customer, left_on="mapped_customer_id", right_on="customer_id", how="right")

        print(map_k(combined["article_id"].tolist(), combined["prediction"].tolist()))
    else:
        submission.drop(columns=["mapped_customer_id"], inplace=True)
        submission["prediction"] = submission["prediction"].apply(lambda val: " ".join(f"0{x}" for x in val))

        submission.to_csv(path, index=False, compression="gzip")
        # TODO: automatically upload to kaggle using their api
