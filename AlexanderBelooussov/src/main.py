# based on https://github.com/radekosmulski/personalized_fashion_recs

import argparse
from functools import partialmethod

from sklearn.model_selection import ParameterGrid

from rank import *
from samples import *

import warnings

warnings.filterwarnings("ignore")


def random_search_grid_cv(runs=10, frac=0.05):
    """
    Random search for hyperparameters
    Warning: takes a long time to run
    :param runs: number of runs
    :param frac: fraction of customers to use
    :return:
    """
    grid = {
        'n_train_weeks': [5, 8, 12],
        'n': [12, 25, 50],
        'l0': [True, False],
        'w2v': [True, False],
        'random': [True, False],
        'itemknn': [True, False],
        'n_estimators': [100, 10, 200, 50],
        'ratio': [1, 10, 30, 100],
        'only_candidates': [True, False],
    }

    grid = list(ParameterGrid(grid))
    # shuffle and take given amount of runs
    np.random.shuffle(grid)
    grid = grid[:runs]

    scores = []
    recalls = []
    for params in grid:
        try:
            score, recall = main(
                n_train_weeks=params['n_train_weeks'],
                n=params['n'],
                frac=frac,
                ratio=params['ratio'],
                cv=True,
                verbose=False,
                pp_params=None,
                itemknn=params['itemknn'],
                l0=params['l0'],
                w2v=params['w2v'],
                p2v=False,
                random_samples=params['random'],
                n_estimators=params['n_estimators'],
                only_candidates=params['only_candidates']
            )
            print(f"{params.items()}: {score:.4}, {recall:.4}")
            scores.append(score)
            recalls.append(recall)
        except KeyError:
            # not sure why this happens sometimes
            scores.append(0.)
            recalls.append(0.)

    best_score = np.argsort(scores)
    best_recall = np.argsort(recalls)
    print(f"BEST SCORES")
    for i, r in enumerate(best_score):
        print(f"{i} ({scores[r]}, {recalls[r]}): {grid[r]}")

    print(f"\nBEST RECALLS")
    for i, r in enumerate(best_recall):
        print(f"{i} ({recalls[r]}, {scores[r]}): {grid[r]}")


def main(
        n_train_weeks=12,
        n=12,
        frac=0.1,
        ratio=1,
        cv=True,
        verbose=True,
        pp_params=None,
        itemknn=True,
        l0=True,
        w2v=True,
        p2v=False,
        random_samples=True,
        n_estimators=100,
        preprocessed_data=None,
        only_candidates=False,
):
    methods = []
    if itemknn:
        methods.append('itemknn')
    if l0:
        methods.append('l0')
    if w2v:
        methods.append('w2v')
    if p2v:
        methods.append('p2v')
    if random_samples:
        methods.append('random')
    print(
        f"Running with: frac={frac}, n={n}, n_train_weeks={n_train_weeks}, ratio={ratio}, cv={cv}, verbose={verbose}, "
        f"methods={methods}, n_estimators={n_estimators}")

    if pp_params is None:
        pp_params = {'w2v_vector_size': 25}

    write_sub = not cv and frac == 1.0

    if preprocessed_data is None:
        # load data
        data_dict = load_data(verbose=verbose, frac=frac, seed=42)

        if cv:  # make train and validation sets
            transactions_train, transactions_val = test_train_split(data_dict['transactions'])
            data_dict['transactions'] = transactions_train
            data_dict['transactions_val'] = transactions_val
            transactions_val = customer_id_transform(data_dict['transactions_val'], "customer_id")
            data_dict['transactions_val'] = transactions_val

        # preprocess data
        data_dict = pp_data(data_dict, force=True, write=False, verbose=verbose, params=pp_params)
    else:
        data_dict = preprocessed_data

    # generate samples and candidates
    data_dict = samples(data_dict, n_train_weeks=n_train_weeks, n=n, ratio=ratio, methods=methods, verbose=verbose,
                        only_candidates=only_candidates)

    if cv:
        train_customers = data_dict['transactions']['customer_id'].unique()
        val_customers = data_dict['transactions_val']['customer_id'].unique()
        print(f"#Customers in train: {len(train_customers)}")
        print(f"#Customers in val: {len(val_customers)}")
        n_intersect = len(set(val_customers) - set(train_customers))
        print(
            f"#Customers in val but not in train: {n_intersect}, {n_intersect / len(val_customers) * 100:.2f}% of val")

    # train model and make predictions
    predictions = rank(data_dict, verbose=verbose, n_estimators=n_estimators)

    if cv:  # evaluate predictions
        map_score = map_at_12(predictions, data_dict['transactions_val'])
        print(f"MAP@12: {map_score:.4f}")
        recall = candidates_recall_test(data_dict, data_dict['transactions_val'])
        print(f"RECALL of candidates: {recall:.4f}")
        return map_score, recall

    if write_sub:  # write submission
        # transform customer_id back to original
        predictions.rename(columns={'customer_id': 'transformed'}, inplace=True)
        predictions = predictions.merge(data_dict['customer_keys'], how='left', on='transformed')
        predictions.drop(columns=['transformed'], inplace=True)
        # write
        write_submission(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train_weeks', type=int, default=12)
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--cv', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--itemknn', action='store_true', default=False)
    parser.add_argument('--l0', action='store_true', default=False)
    parser.add_argument('--w2v', action='store_true', default=False)
    parser.add_argument('--p2v', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--grid', action='store_true', default=False)
    parser.add_argument('--only_candidates', action='store_true', default=False)

    args = parser.parse_args()

    if not args.verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    if args.grid:
        random_search_grid_cv(runs=15, frac=args.frac)
    else:
        main(
            n_train_weeks=args.n_train_weeks,
            n=args.n,
            frac=args.frac,
            ratio=args.ratio,
            cv=args.cv,
            verbose=args.verbose,
            itemknn=args.itemknn,
            l0=args.l0,
            w2v=args.w2v,
            p2v=args.p2v,
            random_samples=args.random,
            n_estimators=args.n_estimators,
            only_candidates=args.only_candidates
        )
