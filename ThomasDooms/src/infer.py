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
    # This may look complicated, because it is!
    cor = list(accumulate([int(r == t) for r, t in zip(predicted, actual)]))
    return sum((r == t) * c / (i + 1) for i, (r, t, c) in enumerate(zip(predicted, actual, cor))) / min(len(actual), 12)


def map_k(actual, predicted):
    return sum(ap_k(a, p) for a, p in zip(actual, predicted)) / len(actual)


def compute_baseline(transactions, test_week):
    return transactions[transactions["week"] == test_week - 1]["article_id"].value_counts().head(12).index.tolist()


def infer(model, test, submission, baseline, columns, path, cv):
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
