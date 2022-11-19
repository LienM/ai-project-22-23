import os.path
import random
import pandas as pd


from lightgbm import LGBMRanker


def split_test_train(data, test_week):
    data.sort_values(["week", "customer_id"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    train = data[data["week"] != test_week]
    test = data[data["week"] == test_week].drop_duplicates(["customer_id", "article_id", "sales_channel_id"]).copy()

    groups = train.groupby(["week", "customer_id"])["article_id"].count().values

    return train, test, groups


def train_model(train_x, train_y, groups):
    seed = 42069  # picked by pure random chance

    params = {
        "objective": "binary",
        "boosting": "gbdt",
        "max_depth": -1,
        "num_leaves": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "bagging_seed": seed,
        "learning_rate": 0.05,
        "feature_fraction": 0.6,
        "min_data_in_leaf": 100,
        "lambda_l1": 0,
        "lambda_l2": 0,
        "random_state": seed,
        "metric": "auc",
        "verbose": -1
    }

    ranker = LGBMRanker(**params)

    ranker = ranker.fit(
        train_x,
        train_y,
        group=groups,
    )

    return ranker


def main():
    random.seed(42)

    size = "all"
    recreate = False
    cv = False

    transactions = pd.read_feather(f"data/transactions_{size}.feather")
    articles = pd.read_feather("data/articles.feather")
    customers = pd.read_feather("data/customers.feather")

    max_week = transactions.week.max()
    test_week = max_week + int(not cv)  # increase by 1 if not using cross-validation
    transactions = transactions[transactions["week"] > max_week - 10 - cv]

    path = f"data/samples_{size}.feather"
    if recreate or not os.path.exists(path):
        print("Creating new samples")
        last_purchases = last_purchase_candidates(transactions, test_week)
        bestsellers, previous_bestsellers = bestseller_candidates(transactions, test_week)

        transactions["purchased"] = 1

        data = pd.concat([transactions, last_purchases, bestsellers])
        data.fillna({"purchased": 0}, inplace=True)
        data["purchased"] = data["purchased"].astype("int8")

        data.drop_duplicates(["customer_id", "article_id", "week"], inplace=True)

        data = pd.merge(
            data,
            previous_bestsellers[['week', 'article_id', 'bestseller_rank']],
            on=['week', 'article_id'],
            how='left'
        )

        data = data[data.week != data.week.min()]
        data.bestseller_rank.fillna(999, inplace=True)

        data.reset_index(drop=True, inplace=True)
        data.to_feather(path)
        print(f"Done generating samples {data.size}")
    else:
        data = pd.read_feather(path)
        print(f"Using existing samples {data.size}")

    data = pd.merge(data, articles, on="article_id", how="left")
    data = pd.merge(data, customers, on="customer_id", how="left")

    train, test, groups = split_test_train(data, test_week)

    columns = ["article_id", "product_type_no", "graphical_appearance_no", "colour_group_code",
               "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code",
               "index_group_no", "section_no", "garment_group_no", "FN", "Active", "club_member_status",
               "fashion_news_frequency", "age", "postal_code", "bestseller_rank"]

    columns += [f"prod_name_{i}" for i in range(16)]
    columns += [f"detail_desc_{i}" for i in range(16)]

    ranker = train_model(train[columns], train["purchased"], groups)

    for i in ranker.feature_importances_.argsort()[::-1][:20]:
        print(columns[i], ranker.feature_importances_[i] / ranker.feature_importances_.sum())

    test["prediction"] = ranker.predict(test[columns])
    test.sort_values(["customer_id", "prediction"], ascending=False, inplace=True)
    predicted = test.groupby("customer_id")["article_id"].apply(list).to_dict()

    submission = pd.read_feather("data/example.feather")

    baseline = transactions[transactions["week"] == test_week - 1]["article_id"].value_counts().head(12).index.tolist()

    submission["c_id_mapped"] = submission["customer_id"].map(lambda x: int(x[-16:], 16)).astype("int32")
    submission["prediction"] = submission["c_id_mapped"].apply(lambda x: (predicted.get(x, []) + baseline)[:12])

    if cv:
        per_customer = test[test["purchased"] == 1].groupby("customer_id")["article_id"].apply(list)
        combined = pd.merge(submission, per_customer, left_on="c_id_mapped", right_on="customer_id", how="right")
        print(map_k(combined["article_id"].tolist(), combined["prediction"].tolist()))

    submission.drop(columns=["c_id_mapped"], inplace=True)
    submission["prediction"] = submission["prediction"].apply(lambda val: " ".join(f"0{x}" for x in val))

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    pd.options.display.max_columns = None
    pd.options.display.width = None
    main()
