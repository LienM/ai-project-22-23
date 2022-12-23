import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from BasilRommens.helper.globals import usable_cols
from BasilRommens.helper.dataset import read_data_set, part_data_set, \
    impute_age, get_relevant_cols, construct_X_y
from BasilRommens.notebook.notebook import last_n_week_transactions, \
    merge_transactions, age_bin_candidates, get_group_sizes, \
    show_feature_importances, get_candidate_bestsellers
from BasilRommens.notebook.predictions import make_and_write_predictions_age, \
    make_and_write_predictions


def age_smarter_bins_article():
    """
    make predictions for smart age bins created per week for the article id
    :return: nothing
    """
    # add appropriate column for bestselling rank
    usable_cols.append('age_bestseller_rank')

    for smart_threshold in range(10, 200, 20):
        print(f'age smarter bins articles with threshold {smart_threshold}')
        articles, customers, transactions = read_data_set('feather')
        # articles, customers, transactions = part_data_set('01')

        # encode and decode customer ids
        customer_encoder = joblib.load('data/customer_encoder.joblib')

        articles, customers, transactions = get_relevant_cols(articles,
                                                              customers,
                                                              transactions)

        last_week = 106
        test_week = last_week + 1

        # get transactions in the last n weeks
        transactions = last_n_week_transactions(transactions, last_week, 10)

        # label the ordered columns
        transactions['ordered'] = 1

        # combine the transactions dataframe with all the articles
        transactions = merge_transactions(transactions, articles, customers)
        del articles

        # change all the -1 ages to the most similar age using purchases
        transactions = impute_age(transactions)

        # from here all the code has been used from unless marked otherwise
        # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
        # get the candidates last purchase
        # OWN: get the candidates age bin
        intervals = 'smart'
        candidates_age_bin, age_bestsellers_previous_week = \
            age_bin_candidates(transactions, test_week, intervals=intervals,
                               group_type='article_id')

        # add the non-ordered items
        transactions = pd.concat([transactions, candidates_age_bin])
        transactions['ordered'] = transactions['ordered'].fillna(0)

        transactions = transactions.drop_duplicates(
            subset=['customer_id', 'article_id', 'week'], keep=False)

        transactions['ordered'] = transactions['ordered'].astype(np.uint8)

        # merge the previous week bestsellers to transactions as it adds the rank
        transactions = pd.merge(
            transactions,
            age_bestsellers_previous_week[
                ['week', 'age_bin', 'article_id', 'age_bestseller_rank']],
            on=['week', 'age_bin', 'article_id'],
            how='left'
        )

        # take only the transactions which aren't in the first week
        transactions = transactions[
            transactions['week'] != transactions['week'].min()]

        # change the values for the non-bestsellers to 999
        transactions['age_bestseller_rank'].fillna(999, inplace=True)

        # sort the transactions by week and customer_id
        transactions = transactions \
            .sort_values(['week', 'customer_id']) \
            .reset_index(drop=True)

        # take the training set as the transactions not happening in the test week
        train = transactions[transactions.week != test_week]
        # create the test week
        test = transactions[transactions.week == test_week] \
            .drop_duplicates(
            ['customer_id', 'article_id', 'sales_channel_id']) \
            .copy()

        # construct a suitable X and y (where y indicates ordered or not)
        X_train, y_train, = construct_X_y(train, usable_cols)
        X_test, _ = construct_X_y(test, usable_cols)

        # create a model
        group_sizes = get_group_sizes(train)
        model = lgb.LGBMRanker(objective='lambdarank',
                               metric='ndcg',
                               n_estimators=100,
                               importance_type='gain',
                               force_row_wise=True)
        model = model.fit(X=X_train, y=y_train, group=group_sizes)

        # show feature importances
        show_feature_importances(model)

        # make and write predictions
        sub_name = f'age_bin_week_{last_week}_smart_threshold_{smart_threshold}'
        make_and_write_predictions_age(model, test, X_test,
                                       age_bestsellers_previous_week,
                                       customers, customer_encoder,
                                       transactions, last_week, sub_name, False)

    usable_cols.pop()


def age_smarter_bins():
    """
    make predictions for smart age bins created per week using the product type
    no
    :return: nothing
    """
    # add appropriate column for bestselling rank
    usable_cols.append('age_bestseller_rank')

    for smart_threshold in range(10, 200, 20):
        print(f'age smarter bins product type with threshold {smart_threshold}')
        articles, customers, transactions = read_data_set('feather')
        # articles, customers, transactions = part_data_set('01')

        # encode and decode customer ids
        customer_encoder = joblib.load('data/customer_encoder.joblib')

        articles, customers, transactions = get_relevant_cols(articles,
                                                              customers,
                                                              transactions)

        last_week = 106
        test_week = last_week + 1

        # get transactions in the last n weeks
        transactions = last_n_week_transactions(transactions, last_week, 10)

        # label the ordered columns
        transactions['ordered'] = 1

        # combine the transactions dataframe with all the articles
        transactions = merge_transactions(transactions, articles, customers)
        del articles

        # change all the -1 ages to the most similar age using purchases
        transactions = impute_age(transactions)

        # from here all the code has been used from unless marked otherwise
        # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
        # get the candidates last purchase
        # OWN: get the candidates age bin
        intervals = 'smart'
        candidates_age_bin, age_bestsellers_previous_week = \
            age_bin_candidates(transactions, test_week, intervals=intervals,
                               group_type='product_type_no')

        # add the non-ordered items
        transactions = pd.concat([transactions, candidates_age_bin])
        transactions['ordered'] = transactions['ordered'].fillna(0)

        transactions = transactions.drop_duplicates(
            subset=['customer_id', 'product_type_no', 'week'], keep=False)

        transactions['ordered'] = transactions['ordered'].astype(np.uint8)

        # merge the previous week bestsellers to transactions as it adds the rank
        transactions = pd.merge(
            transactions,
            age_bestsellers_previous_week[
                ['week', 'age_bin', 'product_type_no', 'age_bestseller_rank']],
            on=['week', 'age_bin', 'product_type_no'],
            how='left'
        )

        # take only the transactions which aren't in the first week
        transactions = transactions[
            transactions['week'] != transactions['week'].min()]

        # change the values for the non-bestsellers to 999
        transactions['age_bestseller_rank'].fillna(999, inplace=True)

        # sort the transactions by week and customer_id
        transactions = transactions \
            .sort_values(['week', 'customer_id']) \
            .reset_index(drop=True)

        # take the training set as the transactions not happening in the test week
        train = transactions[transactions.week != test_week]
        # create the test week
        test = transactions[transactions.week == test_week] \
            .drop_duplicates(
            ['customer_id', 'article_id', 'sales_channel_id']) \
            .copy()

        # construct a suitable X and y (where y indicates ordered or not)
        X_train, y_train, = construct_X_y(train, usable_cols)
        X_test, _ = construct_X_y(test, usable_cols)

        # create a model
        group_sizes = get_group_sizes(train)
        model = lgb.LGBMRanker(objective='lambdarank',
                               metric='ndcg',
                               n_estimators=100,
                               importance_type='gain',
                               force_row_wise=True)
        model = model.fit(X=X_train, y=y_train, group=group_sizes)

        # show feature importances
        show_feature_importances(model)

        # make and write predictions
        sub_name = f'age_bin_prod_week_{last_week}_smart_threshold_{smart_threshold}'
        make_and_write_predictions_age(model, test, X_test,
                                       age_bestsellers_previous_week,
                                       customers, customer_encoder,
                                       transactions, last_week, sub_name, True)

    usable_cols.pop()


def age_simple_bin_article():
    """
    (correlation analysis)

    make predictions for age bins created per week using the article id
    :return: nothing
    """
    # add appropriate column for bestselling rank
    usable_cols.append('age_bestseller_rank')

    for last_week in [106, 105, 104, 103, 102]:
        for bin_size in [1, 2, 3, 4, 8, 16, 32, 64]:
            print(last_week, bin_size)
            articles, customers, transactions = read_data_set('feather')

            # encode and decode customer ids
            customer_encoder = joblib.load('data/customer_encoder.joblib')

            articles, customers, transactions = get_relevant_cols(articles,
                                                                  customers,
                                                                  transactions)

            test_week = last_week + 1

            # get transactions in the last n weeks
            transactions = last_n_week_transactions(transactions, last_week, 10)

            # label the ordered columns
            transactions['ordered'] = 1

            # combine the transactions dataframe with all the articles
            transactions = merge_transactions(transactions, articles, customers)
            del articles

            # change all the -1 ages to the most similar age using purchases
            transactions = impute_age(transactions)

            # from here all the code has been used from unless marked otherwise
            # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
            # get the candidates last purchase
            # OWN: get the candidates age bin
            intervals = None
            candidates_age_bin, age_bestsellers_previous_week = \
                age_bin_candidates(transactions, test_week, bin_size,
                                   intervals=intervals,
                                   group_type='article_id')

            # add the non-ordered items
            transactions = pd.concat([transactions, candidates_age_bin])
            transactions['ordered'] = transactions['ordered'].fillna(0)

            transactions = transactions.drop_duplicates(
                subset=['customer_id', 'article_id', 'week', 'age_bin'],
                keep=False)

            transactions['ordered'] = transactions['ordered'].astype(np.uint8)

            # merge the previous week bestsellers to transactions as it adds the rank
            transactions = pd.merge(
                transactions,
                age_bestsellers_previous_week[
                    ['week', 'age_bin', 'article_id',
                     'age_bestseller_rank']],
                on=['week', 'age_bin', 'article_id'],
                how='left'
            )

            # take only the transactions which aren't in the first week
            transactions = transactions[
                transactions['week'] != transactions['week'].min()]

            # change the values for the non-bestsellers to 999
            transactions['age_bestseller_rank'].fillna(999, inplace=True)

            # sort the transactions by week and customer_id
            transactions = transactions \
                .sort_values(['week', 'customer_id']) \
                .reset_index(drop=True)

            # take the training set as the transactions not happening in the test week
            train = transactions[transactions.week != test_week]
            # create the test week
            test = transactions[transactions.week == test_week] \
                .drop_duplicates(
                ['customer_id', 'article_id', 'age_bin', 'sales_channel_id']) \
                .copy()

            # construct a suitable X and y (where y indicates ordered or not)
            X_train, y_train, = construct_X_y(train, usable_cols)
            X_test, _ = construct_X_y(test, usable_cols)

            # create a model
            group_sizes = get_group_sizes(train)
            model = lgb.LGBMRanker(objective='lambdarank',
                                   metric='ndcg',
                                   n_estimators=100,
                                   importance_type='gain',
                                   force_row_wise=True)
            model = model.fit(X=X_train, y=y_train, group=group_sizes)

            # show feature importances
            show_feature_importances(model)

            # make and write predictions
            sub_name = f'age_bin_week_{last_week}_bin_size_{bin_size}'
            make_and_write_predictions_age(model, test, X_test,
                                           age_bestsellers_previous_week,
                                           customers, customer_encoder,
                                           transactions, last_week, sub_name,
                                           False)

    usable_cols.pop()


def age_simple_bin():
    """
    make predictions for age bins created per week using the product type no
    :return: nothing
    """
    # add appropriate column for bestselling rank
    usable_cols.append('age_bestseller_rank')

    last_week = 106
    test_week = last_week + 1
    for bin_size in [1, 2, 3, 4, 8, 16, 32, 64]:
        print(bin_size)
        articles, customers, transactions = read_data_set('feather')

        # encode and decode customer ids
        customer_encoder = joblib.load('data/customer_encoder.joblib')

        articles, customers, transactions = get_relevant_cols(articles,
                                                              customers,
                                                              transactions)

        # get transactions in the last n weeks
        transactions = last_n_week_transactions(transactions, last_week, 10)

        # label the ordered columns
        transactions['ordered'] = 1

        # combine the transactions dataframe with all the articles
        transactions = merge_transactions(transactions, articles, customers)
        del articles

        # change all the -1 ages to the most similar age using purchases
        transactions = impute_age(transactions)

        # from here all the code has been used from unless marked otherwise
        # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
        # get the candidates last purchase
        # OWN: get the candidates age bin
        intervals = None
        candidates_age_bin, age_bestsellers_previous_week = \
            age_bin_candidates(transactions, test_week, bin_size,
                               intervals=intervals,
                               group_type='product_type_no')

        # add the non-ordered items
        transactions = pd.concat([transactions, candidates_age_bin])
        transactions['ordered'] = transactions['ordered'].fillna(0)

        transactions = transactions.drop_duplicates(
            subset=['customer_id', 'product_type_no', 'week', 'age_bin'],
            keep=False)

        transactions['ordered'] = transactions['ordered'].astype(np.uint8)

        # merge the previous week bestsellers to transactions as it adds the rank
        transactions = pd.merge(
            transactions,
            age_bestsellers_previous_week[
                ['week', 'age_bin', 'product_type_no', 'age_bestseller_rank']],
            on=['week', 'age_bin', 'product_type_no'],
            how='left'
        )

        # take only the transactions which aren't in the first week
        transactions = transactions[
            transactions['week'] != transactions['week'].min()]

        # change the values for the non-bestsellers to 999
        transactions['age_bestseller_rank'].fillna(999, inplace=True)

        # sort the transactions by week and customer_id
        transactions = transactions \
            .sort_values(['week', 'customer_id']) \
            .reset_index(drop=True)

        # take the training set as the transactions not happening in the test week
        train = transactions[transactions.week != test_week]
        # create the test week
        test = transactions[transactions.week == test_week] \
            .drop_duplicates(
            ['customer_id', 'article_id', 'age_bin', 'sales_channel_id']) \
            .copy()

        # construct a suitable X and y (where y indicates ordered or not)
        X_train, y_train, = construct_X_y(train, usable_cols)
        X_test, _ = construct_X_y(test, usable_cols)

        # create a model
        group_sizes = get_group_sizes(train)
        model = lgb.LGBMRanker(objective='lambdarank',
                               metric='ndcg',
                               n_estimators=100,
                               importance_type='gain',
                               force_row_wise=True)
        model = model.fit(X=X_train, y=y_train, group=group_sizes)

        # show feature importances
        show_feature_importances(model)

        # make and write predictions
        sub_name = f'age_bin_prod_week_{last_week}_bin_size_{bin_size}'
        make_and_write_predictions_age(model, test, X_test,
                                       age_bestsellers_previous_week,
                                       customers, customer_encoder,
                                       transactions, last_week, sub_name, True)

    usable_cols.pop()


def just_popularity():
    """
    make predictions using the popularity metric defined in the original
    notebook
    :return: nothing
    """
    # add appropriate column for bestselling rank
    usable_cols.append('bestseller_rank')
    usable_cols.remove('age_bin')

    articles, customers, transactions = read_data_set('feather')

    # encode and decode customer ids
    customer_encoder = joblib.load('data/customer_encoder.joblib')

    articles, customers, transactions = get_relevant_cols(articles,
                                                          customers,
                                                          transactions)

    last_week = 106
    test_week = 107

    # get transactions in the last n weeks
    transactions = last_n_week_transactions(transactions, last_week, 10)

    # label the ordered columns
    transactions['ordered'] = 1

    # combine the transactions dataframe with all the articles
    transactions = merge_transactions(transactions, articles, customers)
    del articles

    # from here all the code has been used from unless marked otherwise
    # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    # get the candidates last purchase
    candidates_bestsellers, bestsellers_previous_week = \
        get_candidate_bestsellers(transactions, test_week)

    # add the non-ordered items
    transactions = pd.concat(
        [transactions, candidates_bestsellers])
    transactions['ordered'] = transactions['ordered'].fillna(0)

    transactions = transactions.drop_duplicates(
        subset=['customer_id', 'article_id', 'week'], keep=False)

    transactions['ordered'] = transactions['ordered'].astype(
        np.uint8)
    transactions = pd.merge(
        transactions,
        bestsellers_previous_week[
            ['week', 'article_id', 'bestseller_rank']],
        on=['week', 'article_id'],
        how='left'
    )

    # take only the transactions which aren't in the first week
    transactions = transactions[
        transactions['week'] != transactions['week'].min()]

    # change the values for the non-bestsellers to 999
    transactions['bestseller_rank'].fillna(999, inplace=True)

    # sort the transactions by week and customer_id
    transactions = transactions \
        .sort_values(['week', 'customer_id']) \
        .reset_index(drop=True)

    # take the training set as the transactions not happening in the test week
    train = transactions[transactions.week != test_week]
    # create the test week
    test = transactions[transactions.week == test_week] \
        .drop_duplicates(
        ['customer_id', 'article_id', 'sales_channel_id']) \
        .copy()

    # construct a suitable X and y (where y indicates ordered or not)
    X_train, y_train, = construct_X_y(train, usable_cols)
    X_test, _ = construct_X_y(test, usable_cols)

    # create a model
    group_sizes = get_group_sizes(train)
    model = lgb.LGBMRanker(objective='lambdarank',
                           metric='ndcg',
                           n_estimators=100,
                           importance_type='gain',
                           force_row_wise=True)
    model = model.fit(X=X_train, y=y_train, group=group_sizes)

    # show feature importances
    show_feature_importances(model)

    # bestsellers last week
    bestsellers_last_week = \
        bestsellers_previous_week[
            bestsellers_previous_week['week'] == test_week][
            'article_id'].tolist()

    # make and write predictions
    sub_name = f'simply_popularity_week_{last_week}'
    make_and_write_predictions(model, test, X_test, bestsellers_last_week,
                               customers, customer_encoder, sub_name)

    usable_cols.pop()
    usable_cols.append('age_bin')


if __name__ == '__main__':
    age_smarter_bins()
    age_smarter_bins_article()
    age_simple_bin()
    age_simple_bin_article()
    just_popularity()
