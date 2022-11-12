# based on https://github.com/radekosmulski/personalized_fashion_recs

import pandas as pd
from preprocessing import *
from utils import *
from samples import *
from rank import *
from lightgbm.sklearn import LGBMRanker
import argparse


def main(
        n_train_weeks=12,
        n=12,
        frac=0.1,
        cv=True,
        verbose=True,
        pp_params=None
):
    if pp_params is None:
        pp_params = {'w2v_vector_size': 10}
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
    data_dict = samples(data_dict, n_train_weeks=12, n=12)

    # train model and make predictions
    predictions = rank(data_dict)

    if cv:  # evaluate predictions
        print(f"MAP@12: {map_at_12(predictions, data_dict['transactions_val'])}")
        print(
            f"RECALL of candidates: {candidates_recall_test(data_dict, data_dict['transactions_val'], count_missing=True)}")

    if write_sub:  # write submission
        # transform customer_id back to original
        predictions.rename(columns={'customer_id': 'transformed'}, inplace=True)
        predictions = predictions.merge(data_dict['customer_keys'], how='left', on='transformed')
        predictions.drop(columns=['transformed'], inplace=True)
        # write
        write_submission(predictions, 'submission.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train_week', type=int, default=12)
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--cv', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()
    main(
        n_train_weeks=args.n_train_week,
        n=args.n,
        frac=args.frac,
        cv=args.cv,
        verbose=args.verbose
    )
