from sklearn.model_selection import ParameterGrid

from candidates import *
from ranking import *
from preprocessing import *
from samples import *

### TODO LIST
# 1. Feature engineering
# 2. Hyperparameter tuning

default_params = {
    'train_period': 999
}

hyperparameters = {
    'train_period': [999, 7]
}


def grid_search(params):
    articles, customers, transactions = load_data(frac=0.05)
    articles, customers, transactions, cus_keys = pp_data(articles, customers, transactions, force=True, write=False)
    transactions, transactions_val = test_train_split(transactions)

    # list for results
    results = []

    # get all possible combinations of hyperparameters
    combinations = list(ParameterGrid(params))
    for params in combinations:
        print(f"Training with params: {params}")
        # train model
        score = validation_step(articles.copy(), customers, transactions.copy(), transactions_val.copy(), params=params)
        # save results
        results.append((params, score))

    # get best params
    best_params, score = max(results, key=lambda x: x[1])

    print(f"Best params: {best_params}")
    print(f"Best score: {score}")

    return best_params


def validation_step(processed_articles, processed_customers, processed_transactions, validation, params=None):
    if params is None:
        params = default_params
    predictions = make_predictions(processed_articles, processed_customers, processed_transactions, params=params)
    map12 = map_at_12(predictions, validation)
    print(f"MAP@12 score: {map12}")
    return map12


def make_predictions(processed_articles, processed_customers, processed_transactions, params=None):
    if params is None:
        params = default_params
    samples = generate_samples(processed_articles, processed_customers, processed_transactions,
                               force=True, write=False, period=params['train_period'])

    candidates = generate_candidates(
        processed_articles,
        processed_customers,
        processed_transactions,
        method='popular',
        period=7,
        k=300,
    )
    processed_transactions.drop(columns=['t_dat'], inplace=True)

    predictions = {}
    model = lgbm_ranker_train(samples)
    for key in tqdm(candidates, desc='Predicting'):
        prediction = lgbm_ranker_predict(
            model,
            get_data_from_canditates(candidates[key], key, processed_articles, processed_customers,
                                     processed_transactions)
        )
        predictions[key] = prediction
    predictions = dict_to_df(predictions)

    return predictions


def full_run(params=None):
    """
    FOR THE FINAL RESUlT
    :return:
    """
    if params is None:
        params = default_params
    articles, customers, transactions = load_data()
    articles, customers, transactions, cus_keys = pp_data(articles, customers, transactions, force=True)
    predictions = make_predictions(articles, customers, transactions, params=params)
    # replace customer id again
    predictions.rename(columns={'customer_id': 'transformed'}, inplace=True)
    predictions = predictions.merge(cus_keys, how='left', on='transformed')
    predictions.drop(columns=['transformed'], inplace=True)
    write_submission(predictions, append=True if i == 0 else False)


if __name__ == '__main__':
    # get best params
    best_params = grid_search(params=hyperparameters)

    # train on full data set
    # full_run(best_params)

