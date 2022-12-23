import time

import joblib
import numpy as np
import pandas as pd


# file from kaggle (accessed on 22-12-2022)
# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


# file from kaggle (accessed on 22-12-2022)
# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def precision_at_k(top_rel_articles, top_rec_articles, k):
    """
    Computes the precision at k
    :param top_rel_articles: the relevant articles
    :param top_rec_articles: the recommended articles
    :param k: the k to compute precision at
    :return: precision at k
    """
    # check if relevant articles till kth position are in recommended and sum
    count = sum([1 for article in top_rec_articles[:k + 1] if
                 article in top_rel_articles])
    return count / (k + 1)  # + 1 as we start counting from 0


def relevance_at_k(top_rel_articles, top_rec_articles, k):
    """
    Computes the relevance at k
    :param top_rel_articles: relevant articles
    :param top_rec_articles: recommended articles
    :param k: the k to compute relevance at
    :return:
    """
    # check if recommended kth item in relevant items
    if not len(top_rec_articles):
        return 0
    return top_rec_articles[k] in top_rel_articles


def make_rel_rec_list(joined):
    """
    Makes a list of relevant and recommended articles
    :param joined: a joined dataframe where each row is a customer and contains
    the relevant and recommended articles
    :return: the relevant and recommended articles list
    """
    # set up 2 lists for relevant and recommended items
    rel_list = list()  # relevant items
    rec_list = list()  # recommended items

    # iterate over all the customers in the joined dataframe
    for row in joined.iterrows():
        # split and append the string of relevant items to the list
        rel_list.append(
            list(map(lambda x: int(x), row[1]['prediction_rel'].split(' '))))

        # if the recommendation only contains a float then append an empty list
        # as this float is a NaN
        if type(row[1]['prediction_rec']) == float:
            rec_list.append([])
        else:  # do the same as with the relevant items
            rec_list.append(list(
                map(lambda x: int(x), row[1]['prediction_rec'].split(' '))))

    return rel_list, rec_list


def map_at_k(rel, rec, k, kaggle_code=False):
    """
    calculates map@k or mean average precision at k
    :param rel: relevant items in a dataframe
    :param rec: recommended items in a dataframe
    :param k: the k to recommend
    :return: map at k score
    """
    # setting index of rel and rec to customer_id
    rel = rel.set_index('customer_id')
    rec = rec.set_index('customer_id')
    # join both rel and rec
    joined = rel.join(rec, how='left', lsuffix='_rel', rsuffix='_rec')

    rel_list, rec_list = make_rel_rec_list(joined)

    # This code makes use of the kaggle code
    if kaggle_code:
        return mapk(rel_list, rec_list, k)

    # outer sum value
    sum_U = 0
    U_count = 0
    # iterate over all customers to determine map at k scores
    for rec_row in list(joined.iterrows()):
        # add one to count if no recommendations are made
        if rec_row[1]['prediction_rec'] is None or type(
                rec_row[1]['prediction_rec']) == float:
            U_count += 1
            continue

        # get the relevant items for the rec_row
        rel_articles = np.array(rec_row[1]['prediction_rel'].split(' '))
        rel_articles = rel_articles.astype(int)
        rel_articles = list(rel_articles)

        # get the real recommended items for the rec_row
        rec_articles = np.array(rec_row[1]['prediction_rec'].split(' '))
        rec_articles = rec_articles.astype(int)
        rec_articles = list(rec_articles)

        # get the # of recommendations and the # of real relevant articles
        n = len(rec_articles)
        m = len(rel_articles)

        # inner sum value
        sum_k = 0
        # go over all the recommendations
        for at_k in range(min(n, k)):
            p = precision_at_k(rel_articles, rec_articles, at_k)
            relev = relevance_at_k(rel_articles, rec_articles, at_k)
            sum_k += p * relev

        sum_U += sum_k / min(m, k)

        U_count += 1

    return sum_U / U_count


def prepare_transactions(transactions, val_week):
    """
    prepares the transactions for the validation week such that it becomes a
    dataframe with the columns: customer_id and its predictions
    :param transactions: transactions dataframe
    :param val_week: the week to validate on
    :return: dataframe containing predictions and
    """
    # take only transactions from the validation week
    transactions = transactions[transactions['week'] == val_week]

    # sort the transaction by date
    transactions = transactions.sort_values(by=['t_dat'])

    # only take customers that are in the validation week
    customer_ids = transactions['customer_id'].unique()

    # create a dict to convert to a dataframe later on
    c_id_n_a_ids = {'customer_id': list(), 'prediction': list()}

    # iterate over every customer to store the relevant items as a string
    for customer_id in customer_ids:
        # get the articles that the customer bought
        article_ids = transactions[transactions['customer_id'] == customer_id][
                          'article_id'].values[:12]  # 12 is cut-off

        # create a string of the relevant article ids
        article_ids_string = ' '.join(map(lambda x: '0' + str(x), article_ids))

        # store the string and the customer id
        c_id_n_a_ids['prediction'].append(article_ids_string)
        c_id_n_a_ids['customer_id'].append(customer_id)

    # encode the customers in the customer id column
    customer_encoder = joblib.load('data/customer_encoder.joblib')
    c_id_n_a_ids['customer_id'] = customer_encoder.inverse_transform(
        c_id_n_a_ids['customer_id'])

    return pd.DataFrame(c_id_n_a_ids)


def show_cv(map_at_k_dict):
    """
    shows the cross validation results stored per key
    :param map_at_k_dict: the dictionary containing the keys along with scores
    :return: nothing
    """
    for key, values in map_at_k_dict.items():
        print(key, np.average(values))


def cv(transactions, last_weeks, second_it_vals, type_name, second_it_name,
       suffix='', kaggle_code=False):
    """
    performs cross validation on the given validation weeks
    :param transactions: all transactions
    :param last_weeks: last weeks used for training
    :param second_it_vals: the values of the second/inner iteration in
    submission generation
    :param type_name: the name of the type of the method employed
    :param second_it_name: the name of the second/inner iteration in submission
    generation
    :param suffix: the suffix to add to the file name before the extension
    :param kaggle_code: whether to use map@k code from kaggle or not
    :return: the map at k dict, where the keys are the values of the
    second/inner iteration
    """
    # removed this from the loop, now results should be different, however this
    # wasn't inside the

    # for each of the second/inner iteration values generate an entry in the
    # dict
    map_at_k_dict = {second_it_val: list()
                     for second_it_val in second_it_vals}

    # iterate over all the validation weeks
    for last_week in last_weeks:
        # fetch the relevant items for the validation week dataframe
        rel = prepare_transactions(transactions, last_week + 1)

        for second_it_val in second_it_vals:
            sub_name = f'{type_name}_week_{last_week}_{second_it_name}_{second_it_val}'
            if len(suffix):  # add a suffix is there is one to add
                sub_name += f'_{suffix}'
            rec = pd.read_csv(f'out/{sub_name}.csv.gz')  # get recommendations

            start = time.time()
            # add a new entry to the map at k dict for the second iteration
            # value
            map_at_k_dict[second_it_val].append(
                map_at_k(rel, rec, 12, kaggle_code))
            print(f'done map_at_k in {time.time() - start} seconds')

    return map_at_k_dict


if __name__ == "__main__":
    transactions = pd.read_feather('data/transactions.feather')[
        ['customer_id', 'article_id', 'week', 't_dat']]

    # correlation analysis
    n_weeks = range(3, 14, 2)
    last_weeks = range(102, 106)
    type_name = 'correlation'
    second_it_name = 'n_weeks'
    cv_dict = cv(transactions, last_weeks, n_weeks, type_name, second_it_name)
    show_cv(cv_dict)

    # simple age bins on articles
    bin_size = [1, 2, 3, 4, 8, 16, 32, 64]
    last_weeks = range(102, 106)
    type_name = 'age_bin'
    second_it_name = 'bin_size'
    cv_dict = cv(transactions, last_weeks, bin_size, type_name, second_it_name)
    show_cv(cv_dict)

    # simple age bins on product type nos
    bin_size = [1, 2, 3, 4, 8, 16, 32, 64]
    last_weeks = range(102, 106)
    type_name = 'age_bin_prod'
    second_it_name = 'bin_size'
    cv_dict = cv(transactions, last_weeks, bin_size, type_name, second_it_name,
                 kaggle_code=True)
    show_cv(cv_dict)

    # temporal graph embeddings
    n_weeks = [1, 2, 3, 5, 10]
    last_weeks = range(102, 106)
    type_name = 'embedding'
    second_it_name = 'nr'
    suffix = 'temporal'
    cv_dict = cv(transactions, last_weeks, n_weeks, type_name, second_it_name,
                 suffix)
    show_cv(cv_dict)


def compare_rankings(ranking_1, ranking_2):
    """
    comparing 2 rankings with the values being the ids to compare expressed in
    ranking. The similarity will be expressed by the sum of the difference of
    item rankings of the second compared to the first. If the item of the second
    ranking isn't present in the first one it will be the length of the maximal
    length of both rankings times 2. Thus, a lower similarity score is a better
    one. This is not normalized.
    :param ranking_1: first ranking
    :param ranking_2: second ranking
    :return: similarity between the 2 rankings
    """
    # the similarity score
    sim = 0

    # iterate over all the items and determine
    for key_2, rank_idx_2 in ranking_2:
        added = False

        # iterate over all the first ranking items to determine
        for key_1, rank_idx_1 in ranking_1:
            # add idx difference if both keys are the same
            if key_1 == key_2:
                sim += abs(rank_idx_1 - rank_idx_2)
                added = True
                break

        # if the item is not present in the first ranking then add the max of
        # lengths of both rankings, this is a very strong penalization of not
        # being in the rank
        if not added:
            sim += max(len(ranking_1), len(ranking_2)) * 2

    return sim
