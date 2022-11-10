import os.path
import random

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, LabelEncoder

pd.options.display.max_columns = None
pd.options.display.width = None


def generate_negative_samples(pool):
    random.seed(42)
    positive_pairs = list(map(tuple, pool[['customer_id', 'article_id']].drop_duplicates().values))

    real_dates = pool["t_dat"].unique()
    real_customers = pool["customer_id"].unique()
    real_articles = pool["article_id"].unique()
    real_channels = pool["sales_channel_id"].unique()

    article_and_price = pool[["article_id", "price"]] \
        .drop_duplicates("article_id") \
        .set_index("article_id") \
        .squeeze()

    num_neg_samples = int(pool.shape[0] * 1.1)

    neg_dates = np.random.choice(real_dates, size=num_neg_samples)
    neg_articles = np.random.choice(real_articles, size=num_neg_samples)
    neg_customers = np.random.choice(real_customers, size=num_neg_samples)
    neg_channels = np.random.choice(real_channels, size=num_neg_samples)
    neg_prices = article_and_price[neg_articles].values

    columns = [neg_dates, neg_customers, neg_articles, neg_prices, neg_channels]
    neg_transactions = pd.DataFrame(columns, index=pool.columns).T

    df = neg_transactions[~neg_transactions.set_index(["customer_id", "article_id"]).index.isin(positive_pairs)]
    return df.sample(pool.shape[0])


def sample_positive_negative(articles, customers, transactions):
    # assign the positive transactions and generate the negative samples
    positive = transactions
    negative = generate_negative_samples(positive)

    # set the purchased flag to 1 for the positive transactions
    positive['purchased'] = 1
    negative['purchased'] = 0

    # concat the positive and negative transactions and merge with articles and customers
    temp = pd.concat([positive, negative])
    temp = temp.merge(customers, how="inner", on='customer_id')
    temp = temp.merge(articles, how="inner", on='article_id')
    return temp


def eval_candidates(customer, candidate_ids, model, articles, columns):
    # make a dataframe from the ids and join the candidates with the articles
    candidates = pd.DataFrame(candidate_ids, columns=["article_id"])
    candidates = candidates.merge(articles, how="inner", on='article_id')

    # create a dataframe from the customer and cross it with the candidates
    df = customer.to_frame().T.join(candidates, how="cross")

    # add the features to the merged articles and customers
    df = compute_features(df)

    # compute the best 12 predictions with the model
    df['prediction'] = model.predict(df[columns])
    return df.sort_values("prediction", ascending=False)["article_id"].head(12).tolist()


def generate_predictions(articles, customers, transactions, model, columns):
    purchased = transactions.groupby("article_id").size().reset_index(name="purchases")
    baseline = purchased.nlargest(100, "purchases")["article_id"].values

    customers["predictions"] = customers.apply(lambda x: eval_candidates(x, baseline, model, articles, columns), axis=1)
    return customers


def compute_features(transactions):
    # transactions['price'] = pd.DataFrame(minmax_scale(transactions['price']))
    # transactions['age'] = pd.DataFrame(minmax_scale(transactions['age']))

    transactions['FN'] = LabelEncoder().fit_transform(transactions['FN'])
    transactions['Active'] = LabelEncoder().fit_transform(transactions['Active'])
    transactions['club_member_status'] = LabelEncoder().fit_transform(transactions['club_member_status'])
    transactions['fashion_news_frequency'] = LabelEncoder().fit_transform(transactions['fashion_news_frequency'])

    # purchased = transactions[transactions["purchased"] & (transactions["t_dat"] > "2020-08-01")] \
    #     .groupby("article_id") \
    #     .size() \
    #     .reset_index(name="purchases")

    filtered = transactions[transactions["purchased"] == 1] if 'purchased' in transactions else transactions
    purchased = filtered.groupby("article_id").size().reset_index(name="purchases")
    transactions = pd.merge(transactions, purchased, on='article_id', how='left')

    transactions.fillna(0, inplace=True)
    # transactions = pd.get_dummies(transactions, columns=['sales_channel_id'])
    return transactions


def train_model(x, y, report=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'force_row_wise': True
    }

    callback = lgb.early_stopping(stopping_rounds=5)
    model = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=[lgb_eval], callbacks=[callback])

    if not report:
        return model

    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    print(classification_report(y_test, y_pred > 0.5))

    # test = mean_squared_error(y_test, y_pred) ** 0.5
    # print(f'The RMSE of prediction is: {test}')

    return model


def read_csv(which, sample="01"):
    path = f"data/{which}_sample{sample}.csv.gz" if sample else f"data/{which}.csv"

    if not os.path.exists(path) and sample is None:
        return pd.read_csv(path)
    if not os.path.exists(path):
        frac = "0." + sample[1:] if sample[0] == 0 else sample
        result = pd.read_csv(f"data/{which}.csv").sample(frac=frac)
        result.to_csv(path, index=False, compression="gzip")
        return result

    print(f"Reading {path}")
    return pd.read_csv(path)


def read_merged(articles, customers, transactions, sample="01"):
    path = f'data/merged_sample{sample}.csv.gz' if sample else 'data/merged.csv'

    if not os.path.exists(path):
        print("Generating positive/negative samples")
        merged = sample_positive_negative(articles, customers, transactions)
        merged.to_csv(path, index=False, compression="gzip")
    else:
        print("Loading positive/negative samples")
        merged = pd.read_csv(path)

    print("Done loading samples")
    return merged


def read_model(merged, columns, sample="01"):
    path = f'models/gmb{sample}.txt' if sample else 'models/gmb.txt'

    if not os.path.exists(path):
        print("Computing model")
        temp = compute_features(merged)
        gbm = train_model(temp[columns], temp["purchased"])
        gbm.save_model(path)
    else:
        print("Loading model")
        gbm = lgb.Booster(model_file=path)

    print("Done loading model")
    return gbm


def main():
    columns = ['age', 'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'purchases']

    transactions = read_csv("transactions", "01")
    customers = read_csv("customers", None)
    articles = read_csv("articles", None)

    merged = read_merged(articles, customers, transactions, "01")
    model = read_model(merged, columns, "01")

    # predictions = generate_predictions(articles, customers, transactions, model, columns)
    #
    # # set the list of predictions to be a string of space separated values
    # predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(map(str, x)))
    # predictions[["customer_id", "predictions"]].to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()
