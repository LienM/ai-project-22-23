# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import pandas as pd
import time

from paths import path


def index_or_test_week(weeks, test_week):
    try:
        return weeks[weeks.index(test_week) + 1]
    except ValueError:
        return test_week


def last_purchases(transactions, test_week):
    # We want to shift the week of the last purchase to the next week teh customer bought something
    # This way we can use the last purchase as a candidate
    candidates = transactions.sort_values(['customer_id', 'week'])

    candidates['week'] = candidates.drop_duplicates(['customer_id', 'week'])\
        .groupby('customer_id')['week']\
        .shift(-1, fill_value=test_week)\
        .reindex(candidates.index)\
        .groupby(candidates['customer_id'])\
        .ffill()

    candidates["method"] = 2
    return transactions


def bestsellers(transactions, test_week, k=12):
    # Get the mean price for each article for each week to use later
    mean_price = transactions.groupby(['week', 'article_id'])['price'].mean()

    # Gather the statistics for the purchase amount of each article
    sales = transactions \
        .groupby("week")["article_id"].value_counts() \
        .groupby("week").rank(method="dense", ascending=False) \
        .groupby("week").head(k).rename("bestseller").astype("int8")

    # Set the week of the bestselling candidates to next week
    # Every week is based on the bestseller of the previous week
    sales = pd.merge(sales, mean_price, on=["week", "article_id"]).reset_index()
    sales["week"] += 1

    # Take a single transaction per customer per week and merge with the bestsellers
    unique = transactions.groupby(["week", "customer_id"]).head(1).drop(columns=["article_id", "price"]).copy()
    candidates = pd.merge(unique, sales, on="week")

    # Keep a single customer_id for the test_week
    test_set = unique.drop_duplicates("customer_id").reset_index(drop=True)
    test_set["week"] = test_week

    candidates_test_week = pd.merge(test_set, sales, on="week")

    # Concat the training and test candidates
    candidates = pd.concat([candidates, candidates_test_week], copy=False)
    candidates.drop(columns="bestseller", inplace=True)

    candidates["method"] = 1
    return candidates


def generate_candidates(transactions, test_week):
    # It's necessary to concat transactions first, otherwise real transactions will be dropped in deduplication
    candidates = [transactions]

    start = time.time()
    candidates += [bestsellers(transactions, test_week)]
    print(f"done generating bestseller candidates in {time.time() - start:.2f} seconds\n\n")

    start = time.time()
    candidates += [last_purchases(transactions, test_week)]
    print(f"done generating last purchases candidates in {time.time() - start:.2f} seconds\n\n")

    # Assign a positive value to the real transactions
    transactions["purchased"] = 1
    transactions["method"] = 0

    # Concatenate all the real transactions and generated candidates
    start = time.time()
    transactions = pd.concat(candidates, copy=False)
    print(f"concat {time.time() - start:.2f} seconds")

    # Assign a negative value to the candidates
    transactions.fillna({"purchased": 0}, inplace=True)
    transactions["purchased"] = transactions["purchased"].astype("int8")

    # before = list(transactions["method"].value_counts())

    # Remove the duplicates, the real transactions are at the front, so they aren't removed
    start = time.time()
    transactions.drop_duplicates(["customer_id", "article_id", "week"], keep='first', inplace=True)
    print(f"dedup {time.time() - start:.2f} seconds\n")

    # after = list(transactions["method"].value_counts())
    # purchases = list(transactions["purchased"].value_counts())[0]
    #
    # stats = {
    #     "method": ["bestseller"],
    #     "before": before[:-1],
    #     "after": after[:-1],
    #     "diff": [b - a for b, a in zip(before, after)][:-1],
    #     "recall": [f"{100 * (b - a) / purchases:.2f}%" for b, a in zip(before, after)][:-1]
    # }
    # last = {
    #     "method": "total",
    #     "before": sum(before[:-1]),
    #     "after": sum(after[:-1]),
    #     "diff": sum(before[:-1]) - sum(after[:-1]),
    #     "recall": f"{100 * (sum(before[:-1]) - sum(after[:-1])) / purchases:.2f}%"
    # }
    #
    # df = pd.DataFrame(stats)
    # df = df.append(last, ignore_index=True)
    # print(df)

    transactions.drop(columns="method", inplace=True)
    transactions.reset_index(drop=True, inplace=True)

    return transactions


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.width = None
    # pd.options.display.max_rows = None

    test = False

    data = pd.read_feather(path('transactions', 'selected' if test else 'features'))
    data = data[data["week"] > 104 - 10]
    candy = generate_candidates(data, 105)
    candy.to_feather(path('candidates', 'sample' if test else 'full'))
