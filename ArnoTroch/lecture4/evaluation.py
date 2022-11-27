import numpy as np
import pandas as pd
from utils import DATA_PATH


def apk(actual, predicted, k=12):
    """compute the average precision at k.
    source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def evaluate(sub_csv, skip_cust_with_no_purchases=True):
    """given a submission file, return the mean average precision at k
    source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb 
    """
    sub = pd.read_csv(sub_csv)
    validation_set = pd.read_parquet(f'{DATA_PATH}/validation_ground_truth.parquet')

    apks = []

    no_purchases_pattern = []
    for pred, labels in zip(sub.prediction.str.split(), validation_set.prediction.str.split()):
        if skip_cust_with_no_purchases and (labels == no_purchases_pattern):
            continue
        apks.append(apk(pred, labels, k=12))
    return np.mean(apks)
