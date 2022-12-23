import numpy as np
import pandas as pd

from BasilRommens.helper.evaluation import compare_rankings
from BasilRommens.helper.helper import make_low_high_interval
from BasilRommens.notebook.popularity import get_top_product_type_interval


def create_age_bins(bin_size, intervals, smart_bin_threshold, transactions):
    """
    creates age bins in the transactions dataframe according to the parameters
    passed
    :param bin_size: the size of age bins
    :param intervals: if none then use bin size, else if low high intervals are
    given then use those, else if string smart is given use the smart binning
    algorithm
    :param smart_bin_threshold: smart binning threshold
    :param transactions: transactions dataframe
    :return:
    """
    if intervals is None:  # use bin size if no intervals are given
        transactions['age_bin'] = transactions['age'] // bin_size
    elif intervals == 'smart':  # use smart interval algorithm
        # set age bins to the current age
        transactions['age_bin'] = transactions['age']

        # create a new transactions dataframe to collect the different age bins
        # created per week
        new_transactions = pd.DataFrame()
        for week in transactions['week'].unique():
            weekly_transactions = transactions[transactions['week'] == week]

            # get the week intervals
            intervals = smart_merge_intervals(weekly_transactions,
                                              smart_bin_threshold)

            # bin the ages into the intervals
            weekly_transactions['age_bin'] = \
                weekly_transactions['age_bin'] \
                    .apply(get_age_interval_idx, args=(intervals,))

            # extend the new transactions with the bin transformation
            new_transactions = pd.concat(
                [new_transactions, weekly_transactions])

        # set the transactions to the new transactions
        transactions = new_transactions
        del new_transactions
    else:  # use intervals if they are given
        transactions['age_bin'] = transactions['age']
        transactions['age_bin'] = transactions[['age_bin', 'week']] \
            .apply(get_age_interval_idx, args=(intervals,))

    # set the age bin column to uint8 to save space
    transactions['age_bin'].astype(np.uint8)

    return transactions


def smart_merge_intervals(transactions, threshold=100):
    """
    merge ages that are similar to each other in single intervals and return
    :param transactions: transactions dataframe
    :param threshold: the threshold to use for determining the similarity
    :return: return the merged intervals
    """
    # get the product type counts per product type no for an age
    count = transactions \
        .groupby('age')['product_type_no'].value_counts() \
        .rename('count') \
        .reset_index().merge(transactions, on=['age', 'product_type_no'])

    intervals = list()  # list of intervals
    cur_interval = list()  # current interval
    # threshold = 100 forgot to remove

    # a list of similarities
    sims = list()

    # iterate over the ages to determine if subsequent interval with age can be
    # merged
    for age in transactions['age'].unique():
        if age == -1:  # skip if default value
            continue
        if not cur_interval:  # if there isn't any interval then create a new
            cur_interval.append(age)
            continue

        # append age to current interval if age is compatible with cur_interval
        # set
        # popular product types for current interval
        interval_ranking = get_top_product_type_interval(count, cur_interval)
        # popular product types for current age
        age_ranking = get_top_product_type_interval(transactions, [age])

        # calculate the similarity between both rankings
        sim = compare_rankings(interval_ranking, age_ranking)

        # add similarity to list
        sims.append(sim)

        # if the similarity is below the threshold then append age to current
        if sim < threshold:
            cur_interval.append(age)
        else:  # otherwise add the interval to the new intervals
            intervals.append(cur_interval.copy())
            cur_interval.clear()
            cur_interval.append(age)

    # make low high interval from all intervals
    correct_intervals = list()
    for interval in intervals:
        low_high_interval = make_low_high_interval(interval)
        correct_intervals.append(low_high_interval)

    return correct_intervals


def get_age_interval_idx(age, *intervals):
    """
    get the index of the age interval that the age belongs to
    :param age: the age to get the interval index for
    :param intervals: the age intervals as list of lists with a lower and upper
    bound
    :return: the index of the interval where the age belongs to
    """
    # check if intervals are in list format, if not make them
    if type(intervals[0][0]) != list:
        intervals = [intervals]

    # iterate over all the intervals and determine the interval
    for interval_idx, interval in enumerate(intervals[0]):
        if interval[0] <= age <= interval[1]:
            return interval_idx

    return -1  # default value
