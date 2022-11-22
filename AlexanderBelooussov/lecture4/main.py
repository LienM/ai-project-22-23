# based on https://github.com/radekosmulski/personalized_fashion_recs

import argparse
import warnings

from functools import partialmethod

from rank import *
from samples import *


warnings.filterwarnings("ignore")

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
):
    methods = []
    if itemknn: methods.append('itemknn')
    if l0: methods.append('l0')
    if w2v: methods.append('w2v')
    print(f"Running with: frac={frac}, n={n}, n_train_weeks={n_train_weeks}, ratio={ratio}, cv={cv}, verbose={verbose}, methods={methods}")

    if pp_params is None:
        pp_params = {'w2v_vector_size': 25}
    write_sub = not cv

    # load data
    data_dict = load_data(verbose=verbose, frac=frac)

    if cv:  # make train and validation sets
        transactions, transactions_val = test_train_split(data_dict['transactions'])
        data_dict['transactions'] = transactions
        transactions_val = customer_id_transform(transactions_val, "customer_id")
        data_dict['transactions_val'] = transactions_val

    # preprocess data
    data_dict = pp_data(data_dict, force=True, write=False, verbose=verbose, params=pp_params)

    # generate samples and candidates
    data_dict = samples(data_dict, n_train_weeks=n_train_weeks, n=n, ratio=ratio, methods=methods, verbose=verbose)

    # train model and make predictions
    predictions = rank(data_dict, verbose=verbose)

    if cv:  # evaluate predictions
        map_score = map_at_12(predictions, data_dict['transactions_val'])
        print(f"MAP@12: {map_score}")
        recall = candidates_recall_test(data_dict, data_dict['transactions_val'])
        print(f"RECALL of candidates: {recall}")
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
    parser.add_argument('--cv', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--itemknn', action='store_true', default=False)
    parser.add_argument('--l0', action='store_true', default=False)
    parser.add_argument('--w2v', action='store_true', default=False)

    args = parser.parse_args()

    if not args.verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # some tests for the slides :)
    # for n in [12, 25, 50]:
    #     for n_train_weeks in [12, 26, 52]:
    #         try:
    #             print(f"n: {n}, n_train_weeks: {n_train_weeks}")
    #             main(n_train_weeks=n_train_weeks, n=n, frac=args.frac, cv=True, verbose=False)
    #         except:
    #             print("Failed")
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
    )
