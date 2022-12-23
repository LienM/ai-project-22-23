import itertools

import pandas as pd
from tqdm import tqdm

from BasilRommens.helper.globals import usable_cols

from BasilRommens.notebook.age_bin import create_age_bins


def get_group_sizes(dataset):
    """
    get the product_type_no group sizes per customer in a week
    :param dataset: the dataset to get the group sizes from
    :return: the groupsizes per customer in a week for product type no
    """
    group_sizes = dataset \
        .groupby(['week', 'customer_id'])['product_type_no'] \
        .count().values

    return group_sizes


def get_candidate_bestsellers(transactions, test_week):
    """
    gets bestselling candidates for last week without a ranking along with the
    bestselling candidates per week.

    https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    accessed on 23/12/2022
    :param transactions:
    :param test_week:
    :return:
    """
    # get the average price of each article in a given week
    mean_price = transactions \
        .groupby(['week', 'article_id'])['price'].mean()

    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='dense', ascending=False) \
        .groupby('week').head(12).rename('bestseller_rank').astype('int8')

    # merge the mean price and the sales
    bestsellers_previous_week = pd.merge(sales, mean_price,
                                         on=['week', 'article_id']) \
        .reset_index()
    bestsellers_previous_week['week'] += 1  # move the week 1 week forward

    # get a single transaction per customer per week so that we can merge
    # bestseller candidates with the transactions of customers who made a
    # purchase in a week
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()

    # merge all the candidates with the transactions of customers who made a
    # purchase
    candidates_bestsellers = pd.merge(
        unique_transactions,
        bestsellers_previous_week,
        on='week',
    )

    # remove duplicate customer ids in unique customerid transactions
    test_set_transactions = unique_transactions.drop_duplicates(
        'customer_id').reset_index(drop=True)
    # set the test week for all the customers
    test_set_transactions.week = test_week

    # merge the candidates for the test week
    candidates_bestsellers_test_week = pd.merge(
        test_set_transactions,
        bestsellers_previous_week,
        on='week'
    )

    # add the test week candidates to the rest of the candidates
    candidates_bestsellers = pd.concat(
        [candidates_bestsellers, candidates_bestsellers_test_week])
    # remove the bestseller rank and if the ordered column
    candidates_bestsellers.drop(columns=['bestseller_rank', 'ordered'],
                                inplace=True)

    return candidates_bestsellers, bestsellers_previous_week


def last_purchase_candidates(transactions, test_week):
    """
    get the purchases of the last week and shift them to the next week

    https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03c_Basic_Model_Submission.ipynb
    accessed on 23/12/2022
    :param transactions:
    :return:
    """
    # collect all weeks purchased in by customer
    c2weeks = transactions.groupby('customer_id')['week'].unique()

    # per customer week to shift week dict
    c2weeks2shifted_weeks = dict()

    # iterate over all the customers in each week
    for c_id, weeks in c2weeks.items():
        # make a dict of week to shifted week for customer
        c2weeks2shifted_weeks[c_id] = dict()
        # shift 1 week for weeks of customer
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]

        # handle last week as this is the test week
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

    # create a new transactions dataframe for shifting the weeks
    candidates_last_purchase = transactions.copy()

    # iterate over all the weeks that need to be shifted for each customer
    # and collect them in a list to shift them in one go
    weeks = []
    for c_id, week in zip(transactions['customer_id'], transactions['week']):
        weeks.append(c2weeks2shifted_weeks[c_id][week])

    # shift the weeks
    candidates_last_purchase['week'] = weeks

    return candidates_last_purchase


def get_training_week_candidates(transactions, count_type):
    """
    calculates the training week candidates for each week and returns bestsellers
    of previous week, the bestsellers and unique transactions
    :param transactions: transactions dataframe
    :param count_type: the count type to use for determining the ranking of the
    bestsellers
    :return: bestsellers of previous week, unique week, customer pair
    transactions, bestsellers per previous week merged without the bestseller
    rank
    """
    # group the articles by week and get the counts, take top 12 candidates and
    # add a rank
    bestsellers_previous_week = transactions \
        .groupby(['week', 'age_bin'])[count_type].value_counts() \
        .groupby(['week', 'age_bin']).rank(method='dense', ascending=False) \
        .groupby(['week', 'age_bin']).head(12).rename('age_bestseller_rank') \
        .astype('uint8') \
        .reset_index()
    bestsellers_previous_week.week += 1  # move the week 1 week forward
    # ensure that we have only 1 weekly customer id per week
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=[count_type]) \
        .copy()
    # merge the bestsellers with the unique transactions
    candidates_bestsellers = pd.merge(
        unique_transactions,
        bestsellers_previous_week,
        on=['week', 'age_bin'],
    )
    # remove the bestseller rank column
    candidates_bestsellers.drop(columns=['age_bestseller_rank', 'ordered'],
                                inplace=True)
    return bestsellers_previous_week, candidates_bestsellers, unique_transactions


def test_week_candidates(bestsellers_previous_week, candidates_bestsellers,
                         test_week, unique_transactions):
    """
    gets the candidates for the test week
    :param bestsellers_previous_week: the bestsellers of last week
    :param candidates_bestsellers: the current candidate bestsellers
    :param test_week: the test week nr
    :param unique_transactions: the weekly unique transactions
    :return: the candidates for the test week
    """
    # remove duplicate customer ids in unique customer id transactions
    test_set_transactions = unique_transactions.drop_duplicates(
        'customer_id').reset_index(drop=True)
    # set the test week for all the customers
    test_set_transactions.week = test_week
    # merge the candidates for the test week
    candidates_bestsellers_test_week = pd.merge(
        test_set_transactions,
        bestsellers_previous_week,
        on=['week', 'age_bin']
    )
    # add the test week candidates to the rest of the candidates
    candidates_bestsellers = pd.concat(
        [candidates_bestsellers, candidates_bestsellers_test_week])
    # remove the bestseller rank and if the ordered column
    candidates_bestsellers.drop(columns=['age_bestseller_rank', 'ordered'],
                                inplace=True)
    return candidates_bestsellers


def age_bin_candidates(transactions, test_week, bin_size=1,
                       intervals=None, smart_bin_threshold=100,
                       group_type='article_id'):
    """
    bin and rank the candidates by age for the given group type
    :param transactions: transactions dataframe
    :param test_week: last week to use for training
    :param bin_size: the size of age bins
    :param intervals: if none then use bin size, else if low high intervals are
    given then use those, else if string smart is given use the smart binning
    algorithm
    :param smart_bin_threshold: smart binning threshold
    :return: candidates bestsellers for last week and the previous week
    bestsellers candidates with calculated age bins and given group type
    """
    # create age bins for transactions
    transactions = create_age_bins(bin_size, intervals, smart_bin_threshold,
                                   transactions)

    # training week caninates
    bestsellers_previous_week, candidates_bestsellers, unique_transactions = \
        get_training_week_candidates(transactions, group_type)

    # test week caninates
    candidates_bestsellers = test_week_candidates(bestsellers_previous_week,
                                                  candidates_bestsellers,
                                                  test_week,
                                                  unique_transactions)

    return candidates_bestsellers, bestsellers_previous_week


def show_feature_importances(model):
    """
    show the feature importances of the lightgbm model
    :param model: a lightgbm model
    :return: nothing
    """
    # iterate over every feature in reverse, while showing each column and its
    # importance in percentage of the total importance
    for i in model.feature_importances_.argsort()[::-1]:
        print(usable_cols[i],
              model.feature_importances_[i] / model.feature_importances_.sum())


def last_n_week_transactions(transactions, last_week, n):
    """
    get the transactions of the last n weeks, where we define the last week and
    the number of weeks to go back
    :param transactions: transactions dataframe
    :param last_week: the last week to use for transactions
    :param n: the number of weeks to include in the transactions
    :return: the last n weeks of transactions
    """
    return transactions[transactions.week > last_week - n]


def merge_transactions(transactions, articles, customers):
    """
    merge the transactions with the articles and customers to make a big
    dataframe that is easily usable
    :param transactions: transactions dataframe
    :param articles: articles dataframe
    :param customers: customers dataframe
    :return: the merged dataframe
    """
    # merge the dataframes
    transactions = transactions.merge(articles, on='article_id')
    transactions = transactions.merge(customers, on='customer_id')

    transactions = transactions.reset_index(drop=True)

    return transactions


def get_articles_of_product_type(transactions, product_type_no):
    """
    assumes that the transactions are indexed on product no
    get the articles of a specific product type
    :param transactions: the transactions dataframe to get the article ids from
    :param product_type_no: the product type to get the articles from
    :return: articles belonging to the product type
    """
    # try to get the products
    try:
        products = transactions.loc[product_type_no]
    except KeyError:  # if none exist then return empty list
        return []

    return products['article_id'].tolist()


def get_bestseller_dict_article(age_bestsellers_previous_week, last_week):
    """
    get the bestselling items per age bins in dict form
    :param age_bestsellers_previous_week: the bestsellers of previous week
    :param last_week: the week to pull the bestsellers from
    :return: the bestsellers per age bin using articles
    """
    # get last weeks bestsellers per age bin
    bestsellers_last_week = \
        age_bestsellers_previous_week[
            age_bestsellers_previous_week.week == last_week][
            ['article_id', 'age_bin']]

    # get the unique age bins included
    unique_age_bins = bestsellers_last_week.age_bin.unique()

    # create the bestsellers per age bin in dict form and limit bestsellers to
    # 12 entries per age bin
    bestsellers_age_bin_dict = {age_bin:
                                    list(bestsellers_last_week[
                                             bestsellers_last_week[
                                                 'age_bin'] == age_bin
                                             ]['article_id'].values)[:12]
                                for age_bin in unique_age_bins}

    return bestsellers_age_bin_dict


def get_bestseller_dict(age_bestsellers_previous_week, transactions, last_week):
    """
    get the bestselling items per age bins using product types in dict form
    :param age_bestsellers_previous_week: the bestsellers of previous week
    :param transactions: transactions dataframe
    :param last_week: the week to pull the bestsellers from
    :return: the bestsellers per age bin using product types
    """
    # get last weeks bestsellers per age bin
    bestsellers_last_week = \
        age_bestsellers_previous_week[
            age_bestsellers_previous_week.week == last_week][
            ['product_type_no', 'age_bin']]

    # get the unique age bins
    unique_age_bins = bestsellers_last_week.age_bin.unique()

    # create the bestsellers of product types per age bin in dict form and limit
    # bestsellers to 12 entries per age bin
    bestsellers_age_bin_dict = {age_bin: list(bestsellers_last_week[
                                                  bestsellers_last_week[
                                                      'age_bin'] == age_bin][
                                                  'product_type_no'].values) for
                                age_bin in unique_age_bins}

    # set index for product type no to retrieve this faster
    transactions = transactions.set_index('product_type_no')
    # iterate over age bins to link articles to the product type in order to
    # recommend
    for age_bin in tqdm(unique_age_bins):
        # get the age bin bestsellers
        age_bin_bestsellers = bestsellers_age_bin_dict[age_bin]

        # get the corresponding articles per product type
        age_bin_bestsellers = \
            [get_articles_of_product_type(transactions, product_type_no)
             for product_type_no in age_bin_bestsellers]

        # set the new age bin bestsellers
        new_age_bin_bestsellers = []
        for age_bin_bestseller in age_bin_bestsellers:
            if type(age_bin_bestseller) == list:  # remove duplicates
                new_age_bin_bestsellers.append(list(set(age_bin_bestseller)))
            else:  # add the single bestseller
                new_age_bin_bestsellers.append([age_bin_bestseller])

        # set the new age bin bestsellers by chaining them in one long list
        # and taking only the last 12 entries
        bestsellers_age_bin_dict[age_bin] = \
            list(itertools.chain(*new_age_bin_bestsellers))[:12]

    return bestsellers_age_bin_dict


def get_cid_2_preds(predictions):
    """
    get the customer id to predictions dict from the predictions dataframe, as
    there are multiple customers per article
    :param predictions: predictions per customer dataframe
    :return: customer id to article predictions dict
    """
    return predictions \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()


def get_cid_2_age_bin(transactions):
    """
    get the customer id to age bin dict
    :param transactions: the transactions dataframe
    :return: age bin to which a customer belongs, weeks are not taken into
    account
    """
    return transactions.groupby('customer_id')['age_bin'] \
        .first() \
        .to_dict()
