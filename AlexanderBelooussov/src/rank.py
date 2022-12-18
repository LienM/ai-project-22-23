import lightgbm
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils.class_weight
from lightgbm.sklearn import LGBMRanker


def rank(data, verbose=True, n_estimators=100):
    """
    Create train and test sets for LightGBM Ranker
    Train LightGBM model and make predictions
    :param data:
    :param verbose:
    :param n_estimators:
    :return:
    """
    use_val = False
    ranker_type = "classifier"

    test_week = data['test_week']
    samples = data['samples']
    train = samples[samples.week != test_week].sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    if use_val:
        train = train[train.week != test_week - 1].sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    if verbose:
        print(train.info())
    test = samples[samples.week == test_week].drop_duplicates(['customer_id', 'article_id']).copy()

    val = samples[samples.week == test_week - 1].drop_duplicates(['customer_id', 'article_id']).copy() if use_val else None

    train_baskets = train.groupby(['week', 'customer_id'])['article_id'].count().values
    val_baskets = val.groupby(['week', 'customer_id'])['article_id'].count().values if use_val else None
    columns_to_use = train.columns.difference(['purchased', 'day_of_week', 'month', 'year', 'day'])
    train_x = train[columns_to_use]
    val_x = val[columns_to_use] if use_val else None
    train_y = train['purchased'].astype('int8').tolist()
    val_y = val['purchased'].astype('int8').tolist() if use_val else None

    test_x = test[columns_to_use]
    if ranker_type == "ranker":
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=12,
            boosting_type="dart",
            n_estimators=n_estimators,
            importance_type='gain',
            verbose=10 if verbose else 1,
            class_weight='balanced',
            early_stopping_rounds=20
        )
        ranker = ranker.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y), (train_x, train_y)] if use_val else [(train_x, train_y)],
            group=train_baskets,
            eval_group=[val_baskets, train_baskets] if use_val else [train_baskets],
        )
    elif ranker_type == "classifier":
        ranker = lightgbm.sklearn.LGBMClassifier(
            objective="binary",
            boosting_type="dart",
            n_estimators=n_estimators,
            importance_type='gain',
            verbose=10 if verbose else 1,
            is_unbalance=True,
            early_stopping_rounds=20
        )
        ranker = ranker.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y), (train_x, train_y)] if use_val else [(train_x, train_y)]
        )
    elif ranker_type == "train":
        weights = sklearn.utils.class_weight.compute_sample_weight("balanced", train_y)
        trainset = lightgbm.Dataset(data=train_x, label=train_y, group=train_baskets, weight=weights)
        params = {
            "objective": "lambdarank",
            "boosting": "gbdt",
            "metric": "map",
            "eval_at": [12],
            "n_estimators": n_estimators,
            "class_weight": 'balanced',
            "verbose": 10 if verbose else 1,
            "early_stopping_rounds": 20
        }
        ranker = lightgbm.train(params=params, train_set=trainset, num_boost_round=n_estimators, valid_sets=[trainset])
    else:
        raise ValueError(f"{ranker_type} is not a valid ranker_type")
    if verbose and ranker_type != "train":
        fig, ax = plt.subplots()
        ax = lightgbm.plot_metric(ranker, ax=ax)
        ax.plot()
        plt.plot()
        print(ranker.evals_result_)
        for i in ranker.feature_importances_.argsort()[::-1]:
            print(columns_to_use[i], ranker.feature_importances_[i] / ranker.feature_importances_.sum())

    if ranker_type == "classifier":
        proba = ranker.predict_proba(test_x)
        test['preds'] = [x[1] for x in proba]
    else:
        test['preds'] = ranker.predict(test_x)
    # print(test['preds'].describe())
    # print(test.head(50))

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()

    bestsellers_previous_week = data['article_week_info'][['article_id', 'week', 'bestseller_rank']].sort_values(
        by=['week', 'bestseller_rank'])
    bestsellers_last_week = bestsellers_previous_week[bestsellers_previous_week.week ==
                                                      bestsellers_previous_week.week.max()]['article_id'].tolist()

    customers = data['customers']
    preds = []
    for c_id in customers['customer_id'].unique():
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + bestsellers_last_week
        preds.append(pred[:12])

    predictions = pd.DataFrame({'customer_id': customers['customer_id'].unique(), 'prediction': preds})
    return predictions
