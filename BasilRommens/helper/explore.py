import joblib
import scipy.spatial as scs
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

from BasilRommens.helper.dataset import part_data_set, read_data_set
from BasilRommens.notebook import get_relevant_cols, merge_transactions, \
    change_age


def show_item_count_per_age_bin():
    """
    We use this function to show the distribution of product type no counts per
    age
    :return: nothing
    """
    articles, customers, transactions = read_data_set('feather')

    # taking the relevant columns of all dataframes
    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)

    # combining everything in one dataframe
    transactions = merge_transactions(transactions, articles, customers)
    del articles, customers

    # grouping by age and then finding the total non-zero product type no counts
    transactions = transactions.groupby('age')['product_type_no'] \
        .count() \
        .rename('count') \
        .reset_index()

    # show the distribution of product type counts per age in a barplot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=transactions, x='age', y='count', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    plt.show()


def show_article_id_boxplot(transactions, week, age_bin_size):
    """
    boxplot the number of articles grouped by article id in a given week per age
    bin
    :param transactions: transactions dataframe
    :param week: the week to plot the top n articles for
    :param age_bin_size: the bin size of the age
    :return: nothing
    """
    # get only transactions in the desired week
    transactions = transactions[transactions['week'] == week]

    # create the age bins
    transactions['age_bin'] = transactions['age'] // age_bin_size

    # get the counts of article ids per article_id in an age bin
    transactions = \
        transactions.groupby(['age_bin', 'article_id'])['article_id'] \
            .count() \
            .rename('article_id_count') \
            .reset_index()

    # plot the article id counts in a boxplot format per age bin
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(data=transactions, x='age_bin', y='article_id_count', ax=ax)
    ax.set_xlabel('age bin')
    ax.set_ylabel('count')
    plt.show()


def show_product_type_no_boxplot(transactions, week, age_bin_size):
    """
    boxplot the number of sold product types in a given week no per age bin
    :param transactions: transactions dataframe
    :param week: teh week to plot the top n product types for
    :param age_bin_size: the bin size of the age
    :return: nothing
    """
    # get only transactions in the desired week
    transactions = transactions[transactions['week'] == week]

    # create the age bins
    transactions['age_bin'] = transactions['age'] // age_bin_size

    # get the counts of product type no per product_type_no in an age bin
    transactions = \
        transactions.groupby(['age_bin', 'product_type_no'])['product_type_no'] \
            .count() \
            .rename('product_type_count') \
            .reset_index()

    # plot the product type counts in a boxplot format per age bin
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(data=transactions, x='age_bin', y='product_type_count', ax=ax)
    ax.set_xlabel('age bin')
    ax.set_ylabel('count')
    plt.show()


def get_index_vector(indices, values, size):
    """
    get a vector of size `size` with the values at the indices of the keys, so
    that we can skip certain indices
    :param indices: the indices of the vector
    :param values: the values to shuffle
    :param size: the size of the vector
    :return: the vector
    """
    vec = np.zeros(size)
    vec[indices] = values
    return vec


def get_subsequent_weeks_corr_list(transactions):
    """
    get the correlation between subsequent weeks
    :param transactions: transactions
    :return: the list of correlations
    """
    # get the sold product type no counts per age bin
    transactions = transactions.groupby(['week', 'product_type_no'])[
        'product_type_no'] \
        .count() \
        .rename('product_type_count') \
        .reset_index()

    # add 1 to product type number so that product_type_no is index
    transactions['product_type_no'] += 1
    vec_size = transactions['product_type_no'].max() + 1  # add one to reindex

    # get the vectors of product type counts per week
    vecs = list()
    for week in sorted(transactions['week'].unique()):
        # get the week transactions
        week_transactions = transactions[transactions['week'] == week]

        # get the product type counts and the corresponding product type no
        week_product_type_count = week_transactions[
            ['product_type_count']].values
        week_product_type_no = week_transactions[
            ['product_type_no']].values

        # get the vector of product type counts
        vec = get_index_vector(week_product_type_no, week_product_type_count,
                               vec_size)

        # add the vector to the list of vectors
        vecs.append(vec)

    vecs = np.array(vecs)  # make an array as it simplifies things

    # get line plot of correlation between subsequent weeks
    dist_list = list()
    for i, vec0 in enumerate(vecs):
        if i == len(vecs) - 1:
            break
        vec1 = vecs[i + 1]
        dist = scs.distance.pdist(np.array([vec0, vec1]), 'correlation')[0]
        dist_list.append(dist)
    return dist_list


def show_subsequent_week_corr(transactions):
    """
    plot the correlation between subsequent weeks
    :param transactions: transactions
    :return: nothing
    """
    dist_list = get_subsequent_weeks_corr_list(transactions)
    sns.lineplot(x=range(len(dist_list)), y=dist_list)
    plt.show()


def calc_correlation_stability(transactions):
    """
    calculates the stability of subsequent weeks by summing the correlation
    :param transactions: transactions dataframe
    :return: the stability of subsequent weeks
    """
    return sum(get_subsequent_weeks_corr_list(transactions))


def explore_rq1():
    """
    The full code of exploration of the first research question
    :return: nothing
    """
    # load the data
    articles, customers, transactions = read_data_set('feather')
    # only get the relevant columns
    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)
    # merge the dataframes into one
    transactions = merge_transactions(transactions, articles, customers)

    # comparing the distribution of count of items with product no and count of
    # items with article id
    show_article_id_boxplot(transactions, 106, 1)
    show_product_type_no_boxplot(transactions, 106, 1)
    # what about slightly bigger bins?
    show_article_id_boxplot(transactions, 106, 2)
    show_article_id_boxplot(transactions, 106, 3)
    show_article_id_boxplot(transactions, 106, 4)

    # showing correlation between subsequent weeks for product type no counts
    show_subsequent_week_corr(transactions)

    # calculating the correlation stability per age bin to show the stability
    # between the age bins when making subsequent week purchases, so the lower
    # the graph the more stable the age bin is
    for bin_size in [1, 2, 3, 4]:
        # determine the age bins
        transactions['age_bin'] = transactions['age'] // bin_size

        # calculate the age bin stability
        bin_correlation_stability = list()
        for bin in sorted(transactions['age_bin'].unique()):
            stability = \
                calc_correlation_stability(
                    transactions[transactions['age_bin'] == bin])
            bin_correlation_stability.append(stability)

        # plot a line plot
        sns.lineplot(x=range(len(bin_correlation_stability)),
                     y=bin_correlation_stability) \
            .set(title=f'bin size {bin_size}')
        plt.show()


def explore_age_impute():
    """
    Explore how imputing the age based on correlation of product types bought
    changes the number of products bought per age bin
    :return: nothing
    """
    # load the data
    articles, customers, transactions = read_data_set('feather')
    # get the relevant columns
    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)
    # merge the transactions dataframe to get a better dataframe
    transactions = merge_transactions(transactions, articles, customers)
    del articles, customers

    # age histplot before impute
    ages = transactions['age'].tolist()
    ax = sns.histplot(ages, binwidth=1)
    ax.patches[0].set_facecolor('salmon')
    plt.show()
    print(min(ages), max(ages))

    # impute all the -1 ages with the most similar age based on correlation of
    # number of product types bought
    transactions = change_age(transactions)
    ages = transactions['age'].tolist()
    sns.histplot(ages, binwidth=1)
    plt.show()
    print(min(ages), max(ages))


def same_day_purchase():
    """
    explore the number of purchases made on the same day for a customer
    :return: nothing
    """
    # load the data
    articles, customers, transactions = read_data_set('feather')
    # get the relevant columns of the data
    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions,
                                                          ['t_dat'])
    # combine the transactions dataframe with all the articles
    transactions = merge_transactions(transactions, articles, customers)
    del articles, customers

    # get the number of purchases per day per customer id
    nr_purchases = transactions.groupby(['customer_id', 't_dat'])[
        't_dat'].count().values

    # histogram of nr of purchases per day
    sns.histplot(nr_purchases, binwidth=1)
    plt.yscale('log')  # log scale to see the distribution better
    plt.show()

    # histogram of nr of purchase per day per bin where leftmost bar is where
    # only a single purchase was made
    nr_purchases_diff = list(map(lambda x: x != 1, nr_purchases))
    ax = sns.histplot(nr_purchases_diff, bins=2)
    ax.patches[0].set_facecolor('salmon')  # change the color of the first bin
    plt.show()


if __name__ == '__main__':
    show_item_count_per_age_bin()
    explore_rq1()
    explore_age_impute()
    same_day_purchase()
