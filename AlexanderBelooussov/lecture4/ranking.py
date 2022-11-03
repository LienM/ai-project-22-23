from utils import *
from tqdm import tqdm
from lightgbm.sklearn import LGBMRanker


def random_ranker(candidates, articles, customers, transactions):
    result = {}
    for customer_id in tqdm(customers['customer_id'].values, desc="Ranking", leave=False):
        result[customer_id] = np.random.choice(candidates[customer_id], size=12, replace=False)
    return result


def lgbm_ranker_train(train, params):
    model = LGBMRanker(
        objective="lambdarank",
        metric=params['metric'],
        boosting_type=params['boosting_type'],
        n_estimators=100,
        importance_type='gain',
        n_jobs=7,
    )
    qids_train = train.groupby(['week', 'customer_id'])['article_id'].count().values
    y_train = train['y']
    X_train = train.drop(['y', 'customer_id'], axis=1)
    model.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
    )
    # x_train_cols = list(X_train.columns)
    # print(x_train_cols)
    # for i in model.feature_importances_.argsort()[::-1]:
    #     print(x_train_cols[i], model.feature_importances_[i] / model.feature_importances_.sum())
    return model

def lgbm_ranker_predict(model, candidates):
    # make predictions for candidates
    candidates['prediction'] = model.predict(candidates)
    candidates = candidates.sort_values(by=['prediction'], ascending=False)
    # print(candidates.head(12))
    return candidates['article_id'].head(12).values



