from sklearn.model_selection import ParameterGrid

from candidates import *
from ranking import *
from preprocessing import *
from samples import *

### TODO LIST
# 1. Feature engineering
# 2. Hyperparameter tuning

default_train_params = {
    'train_period': 999,
    'boosting_type': 'dart',
    'metric': 'ndcg'
}
default_preprocessing_params = {
    'w2v_vector_size': 25,
}

train_parameters = {
    'train_period': [999, 52, 12, 8, 6, 2, 1],
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'metric': ['ndcg', 'map'],

}
# train_parameters = {
#     'train_period': [1],
#     'boosting_type': ['gbdt'],
#     'metric': ['ndcg'],
#
# }

preprocess_parameters = {
    'w2v_vector_size': [5, 10, 25],
}

VERBOSE = False


def untangle_param_dict(param_dict):
    # split result back into 2 dicts
    pp = {}
    for keys in default_preprocessing_params.keys():
        pp[keys] = param_dict[keys]
        # remove from param_dict
        del param_dict[keys]
    return pp, param_dict


def grid_search(pp_params, t_params, k=3):
    # list for results
    results = {}
    t_combinations = list(ParameterGrid(t_params))
    pp_combinations = list(ParameterGrid(pp_params))
    for i in range(k):
        og_articles, og_customers, og_transactions = load_data(frac=0.05, verbose=VERBOSE)
        for pp_combination in pp_combinations:
            print(f"Preprocessing with params: {pp_combination}")
            articles, customers, transactions, cus_keys = pp_data(og_articles.copy(), og_customers.copy(), og_transactions.copy(),
                                                                  force=True, write=False, verbose=VERBOSE,
                                                                  vector_size=pp_combination['w2v_vector_size'])
            transactions, transactions_val = test_train_split(transactions)

            # get all possible combinations of hyperparameters
            for params in t_combinations:
                print(f"\tTraining with params: {params}\n\t", end='')
                # train model
                score = validation_step(articles.copy(), customers.copy(), transactions.copy(), transactions_val.copy(), params=params)
                # save results
                p = dict(params)
                p.update(pp_combination)
                if tuple(sorted(p.items())) not in results:
                    results[tuple(sorted(p.items()))] = [score]
                else:
                    results[tuple(sorted(p.items()))].append(score)

    # get best params
    best_params, score = max(results.items(), key=lambda x: np.mean(x[1]))
    best_params = dict(best_params)
    print(f"\n\nBest params: {best_params}")
    print(f"Best score mean: {np.mean(score)}, score: {score}\n\n")

    # show top 4 other results
    results = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for i in range(1, min(len(results), 5)):
        print(f"Result {i}: parameters: {dict(results[i][0])}, mean: {np.mean(results[i][1])}, score: {results[i][1]}")

    return untangle_param_dict(best_params)


def validation_step(processed_articles, processed_customers, processed_transactions, validation, params=None):
    if params is None:
        params = default_train_params
    predictions = make_predictions(processed_articles, processed_customers, processed_transactions, params=params)
    map12 = map_at_12(predictions, validation)
    print(f"MAP@12 score: {map12}")
    return map12


def make_predictions(processed_articles, processed_customers, processed_transactions, params=None):
    if params is None:
        params = default_train_params
    samples = generate_samples(processed_articles, processed_customers, processed_transactions,
                               force=True, write=False, period=params['train_period'], verbose=VERBOSE)

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
    model = lgbm_ranker_train(samples, params=params)
    # candidates = tqdm(candidates, desc='Predicting', leave=False) if VERBOSE else candidates
    for key in tqdm(candidates, desc='Predicting', leave=False) if VERBOSE else candidates:
        prediction = lgbm_ranker_predict(
            model,
            get_data_from_canditates(candidates[key], key, processed_articles, processed_customers,
                                     processed_transactions)
        )
        predictions[key] = prediction
    predictions = dict_to_df(predictions)

    return predictions


def full_run(pp_params=None, t_params=None):
    """
    FOR THE FINAL RESUlT
    :return:
    """
    if t_params is None:
        t_params = default_train_params
    if pp_params is None:
        pp_params = default_preprocessing_params
    articles, customers, transactions = load_data()
    articles, customers, transactions, cus_keys = pp_data(articles, customers, transactions,
                                                          force=True, write=False, verbose=VERBOSE,
                                                          vector_size=pp_params['w2v_vector_size'])
    predictions = make_predictions(articles, customers, transactions, params=t_params)
    # replace customer id again
    predictions.rename(columns={'customer_id': 'transformed'}, inplace=True)
    predictions = predictions.merge(cus_keys, how='left', on='transformed')
    predictions.drop(columns=['transformed'], inplace=True)
    write_submission(predictions, append=False)


if __name__ == '__main__':
    # get best params
    best_pp_params, best_train_params = grid_search(pp_params=preprocess_parameters, t_params=train_parameters, k=2)

    # train on full data set
    # full_run(best_pp_params, best_train_params)

