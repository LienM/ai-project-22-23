import pandas as pd
from lightgbm.sklearn import LGBMRanker


def rank(data):
    """
    Create train and test sets for LightGBM Ranker
    Train LightGBM model and make predictions
    :param data:
    :return:
    """
    test_week = data['test_week']
    samples = data['samples']
    train = samples[samples.week != test_week].sort_values(by=['week', 'customer_id'])
    test = samples[samples.week == test_week].drop_duplicates(['customer_id', 'article_id', 'sales_channel_id']).copy()
    train_baskets = train.groupby(['week', 'customer_id'])['article_id'].count().values
    columns_to_use = train.columns.difference(['customer_id', 'week', 'purchased', 'day_of_week', 'month', 'year', 'day'])
    train_x = train[columns_to_use]
    train_y = train['purchased']

    test_x = test[columns_to_use]

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="map",
        boosting_type="dart",
        n_estimators=100,
        importance_type='gain',
        verbose=10
    )

    ranker = ranker.fit(
        train_x,
        train_y,
        group=train_baskets
    )
    for i in ranker.feature_importances_.argsort()[::-1]:
        print(columns_to_use[i], ranker.feature_importances_[i]/ranker.feature_importances_.sum())

    test['preds'] = ranker.predict(test_x)
    # print(test['preds'].describe())
    # print(test.head(50))

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()

    # for i, key in enumerate(c_id2predicted_article_ids.keys()):
    #     if i > 5: break
    #     print(key, c_id2predicted_article_ids[key])

    bestsellers_previous_week = data['article_week_info'][['article_id', 'week', 'bestseller_rank']]
    bestsellers_last_week = \
        bestsellers_previous_week[bestsellers_previous_week.week == bestsellers_previous_week.week.max()]['article_id'] \
        .tolist()

    customers = data['customers']
    preds = []
    for c_id in customers['customer_id'].unique():
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + bestsellers_last_week
        preds.append(pred[:12])

    predictions = pd.DataFrame({'customer_id': customers['customer_id'].unique(), 'prediction': preds})
    return predictions
