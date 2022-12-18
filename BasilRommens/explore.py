import joblib
import scipy.spatial as scs
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

from BasilRommens.dataset import part_data_set, read_data_set
from BasilRommens.lecture4 import get_relevant_cols, merge_transactions, \
    change_age


def exploration():
    articles, customers, transactions = read_data_set('feather')
    # articles, customers, transactions = part_data_set('5')
    customer_encoder = joblib.load('../data/customer_encoder.joblib')

    # taking the relevant columns of all dataframes
    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)

    transactions = transactions.merge(articles, on='article_id')
    del articles
    transactions = transactions.merge(customers, on='customer_id')
    del customers
    transactions = transactions.reset_index(drop=True)

    transactions = transactions.groupby('age')['product_type'] \
        .count() \
        .rename('count') \
        .reset_index()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=transactions, x='age', y='count', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    plt.show()


def plot_top_n_articles(transactions, week, bin_size):
    transactions = transactions[transactions['week'] == week]
    transactions['age_bin'] = transactions['age'] // bin_size
    transactions = \
        transactions.groupby(['age_bin', 'article_id'])['article_id'] \
            .count() \
            .rename('article_id_count') \
            .reset_index()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(data=transactions, x='age_bin', y='article_id_count', ax=ax)
    ax.set_xlabel('age bin')
    ax.set_ylabel('count')
    plt.show()


def plot_top_n_product_no(transactions, week, bin_size):
    transactions = transactions[transactions['week'] == week]
    transactions['age_bin'] = transactions['age'] // bin_size
    transactions = \
        transactions.groupby(['age_bin', 'product_type_no'])['product_type_no'] \
            .count() \
            .rename('product_type_count') \
            .reset_index()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(data=transactions, x='age_bin', y='product_type_count', ax=ax)
    ax.set_xlabel('age bin')
    ax.set_ylabel('count')
    plt.show()


def get_vector(keys, values, size):
    vec = np.zeros(size)
    vec[keys] = values
    return vec


def plot_delta(transactions, bin, bin_size):
    transactions = transactions.groupby(['week', 'product_type_no'])[
        'product_type_no'] \
        .count() \
        .rename('product_type_count') \
        .reset_index()
    # add 1 to product type number so that product_type_no is index
    transactions['product_type_no'] += 1
    vec_size = transactions['product_type_no'].max() + 1

    vecs = list()
    for week in sorted(transactions['week'].unique()):
        week_transactions = transactions[transactions['week'] == week]
        week_product_type_count = week_transactions[
            ['product_type_count']].values
        week_product_type_no = week_transactions[
            ['product_type_no']].values

        vec = get_vector(week_product_type_no, week_product_type_count,
                         vec_size)
        vecs.append(vec)
    vecs = np.array(vecs)

    # get line plot of delta
    dist_list = list()
    for i, vec0 in enumerate(vecs):
        if i == len(vecs) - 1:
            break
        vec1 = vecs[i + 1]
        dist = scs.distance.pdist(np.array([vec0, vec1]), 'correlation')[0]
        dist_list.append(dist)
    sns.lineplot(x=range(len(dist_list)), y=dist_list).set(
        title=f'{bin} bin, {bin_size} bin size')
    plt.show()

    # get heatmap of delta
    # colormap = 'Blues'
    # dist_mat = scs.distance.squareform(scs.distance.pdist(vecs, 'cityblock'))
    # sns.heatmap(dist_mat, cmap=colormap)
    # plt.show()

    # reduce = UMAP(n_components=2, n_neighbors=5, min_dist=0.9, random_state=42,
    #               metric='correlation')
    # # reduce = PCA(n_components=2)
    # reduced_vecs = reduce.fit_transform(vecs)
    # sns.scatterplot(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1])
    # sns.lineplot(x=reduced_vecs[:, 0], y=reduced_vecs[:, 1])
    # plt.show()


def get_stability(transactions, bin, bin_size):
    transactions = transactions.groupby(['week', 'product_type_no'])[
        'product_type_no'] \
        .count() \
        .rename('product_type_count') \
        .reset_index()
    # add 1 to product type number so that product_type_no is index
    transactions['product_type_no'] += 1
    vec_size = transactions['product_type_no'].max() + 1

    vecs = list()
    for week in sorted(transactions['week'].unique()):
        week_transactions = transactions[transactions['week'] == week]
        week_product_type_count = week_transactions[
            ['product_type_count']].values
        week_product_type_no = week_transactions[
            ['product_type_no']].values

        vec = get_vector(week_product_type_no, week_product_type_count,
                         vec_size)
        vecs.append(vec)
    vecs = np.array(vecs)

    # get line plot of delta
    dist_list = list()
    for i, vec0 in enumerate(vecs):
        if i == len(vecs) - 1:
            break
        vec1 = vecs[i + 1]
        dist = scs.distance.pdist(np.array([vec0, vec1]), 'correlation')[0]
        dist_list.append(dist)
    return sum(dist_list)


def explore_rq1():
    # articles, customers, transactions = part_data_set('5')
    articles, customers, transactions = read_data_set('feather')

    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)

    transactions = transactions.merge(articles, on='article_id')
    del articles
    transactions = transactions.merge(customers, on='customer_id')
    del customers
    transactions = transactions.reset_index(drop=True)

    # comparing the distribution of count of items with product no
    # and count of items with article id
    plot_top_n_articles(transactions, 106, 1)
    plot_top_n_product_no(transactions, 106, 1)
    # what about slightly bigger bins?
    plot_top_n_articles(transactions, 106, 2)
    plot_top_n_articles(transactions, 106, 3)
    plot_top_n_articles(transactions, 106, 4)

    # popular product types in next week difference
    plot_delta(transactions, None, None)
    for bin_size in [1, 2, 3, 4]:
        transactions['age_bin'] = transactions['age'] // bin_size
        bin_stability = list()
        for bin in sorted(transactions['age_bin'].unique()):
            stability = \
                get_stability(transactions[transactions['age_bin'] == bin], bin,
                              bin_size)
            bin_stability.append(stability)

        sns.lineplot(x=range(len(bin_stability)), y=bin_stability) \
            .set(title=f'bin size {bin_size}')
        plt.show()


def change_age_explore():
    articles, customers, transactions = read_data_set('feather')
    # articles, customers, transactions = part_data_set('01')

    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions)

    # combine the transactions dataframe with all the articles
    transactions = merge_transactions(transactions, articles, customers)
    del articles

    # TODO: if time change to stacked barplot for better view where the -1s went
    # age histplot before transformation
    ages = transactions['age'].tolist()
    ax = sns.histplot(ages, binwidth=1)
    ax.patches[0].set_facecolor('salmon')
    plt.show()
    print(min(ages), max(ages))
    # change all the -1 ages to the most similar age using purchases
    transactions = change_age(transactions)
    ages = transactions['age'].tolist()
    sns.histplot(ages, binwidth=1)
    plt.show()
    print(min(ages), max(ages))


def same_day_purchase():
    articles, customers, transactions = read_data_set('feather')
    # articles, customers, transactions = part_data_set('01')

    articles, customers, transactions = get_relevant_cols(articles, customers,
                                                          transactions,
                                                          ['t_dat'])

    # combine the transactions dataframe with all the articles
    transactions = merge_transactions(transactions, articles, customers)
    del articles

    # get the number of purchases per day per customer id
    nr_purchases = transactions.groupby(['customer_id', 't_dat'])[
        't_dat'].count().values

    # histogram of nr of purchases per day
    sns.histplot(nr_purchases, binwidth=1)
    plt.yscale('log')
    plt.show()
    # histogram of nr of purchase per day with bin where first bar is where only
    # one purchase was made
    nr_purchases_diff = list(map(lambda x: x != 1, nr_purchases))
    ax = sns.histplot(nr_purchases_diff, bins=2)
    ax.patches[0].set_facecolor('salmon')
    plt.show()


if __name__ == '__main__':
    # explore_rq1()
    change_age_explore()
    # same_day_purchase()
