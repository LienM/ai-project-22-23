from lightgbm.sklearn import LGBMRanker
import pandas as pd
def rank(train_X,train_y,test_X,test,train_baskets,columns_to_use,LGBMBoostingType,bestsellers_previous_week):
    ranker=LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type=LGBMBoostingType,
        n_estimators=1,
        importance_type='gain',
        verbose=10
    )

    ranker = ranker.fit(
        train_X,
        train_y,
        group=train_baskets,
    )

    for i in ranker.feature_importances_.argsort()[::-1]:
        print(columns_to_use[i], ranker.feature_importances_[i]/ranker.feature_importances_.sum())

    test['preds'] = ranker.predict(test_X)

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()

    bestsellers_last_week = \
        bestsellers_previous_week[bestsellers_previous_week.week == bestsellers_previous_week.week.max()]['article_id'].tolist()

    sub = pd.read_csv('../../data/sample_submission.csv')

    preds = []

    def customer_hex_id_to_int(series):
        return series.str[-16:].apply(hex_id_to_int)

    def hex_id_to_int(str):
        return int(str[-16:], 16)


    for c_id in customer_hex_id_to_int(sub.customer_id):
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + bestsellers_last_week
        preds.append(pred[:12])

    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub.prediction = preds

    sub_name = "submissionRobbeLauwers"
    sub.to_csv(f'../../data/subs/{sub_name}.csv.gz', index=False)
    sub.to_csv(f'../../data/subs/{sub_name}.csv', index=False)
    print("Done")
    print(sub_name)