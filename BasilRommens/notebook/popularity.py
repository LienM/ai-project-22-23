import collections

import numpy as np


def popular_predictor(transactions):
    """
    get the most popular articles in a string format to be used for submission
    :param transactions: transactions dataframe
    :return: string of prediction of most popular items
    """
    # get the most popular items in string format
    top_counts = collections.Counter(transactions.groupby('article_id')[
                                         'article_id'].count().to_dict()) \
        .most_common(12)

    # convert the most popular items in list format
    predictions = list(map(lambda x: '0' + str(x[0]), top_counts))

    return ' '.join(predictions)


def get_n_popular_articles(transactions, n):
    """
    get n most popular articles
    :param transactions: transactions dataframe
    :param n: the number of most popular articles
    :return: a list of the top n articles
    """
    # group by articles
    grouped_articles = transactions.groupby('article_id')['article_id']
    # count the number of times each article is bought
    counted_articles = grouped_articles.count()
    # sort descending to have the top items first
    sorted_articles = counted_articles.sort_values(ascending=False)

    # only take top n articles
    top_n_articles = sorted_articles[:n].index

    return list(top_n_articles)


def predict_topn(X_reduced, model):
    """
    predict the top n articles for each customer using the given model
    :param X_reduced: dataframe containing possible articles and customers
    :param model: the model to infer the articles from
    :return: topn articles for each customer
    """
    # top n will be a dict of customer and their respective recommendations
    topn = dict()
    customers = X_reduced['customer_id'].unique()
    articles = X_reduced['article_id'].unique()

    # iterate over all the customers
    for uid in customers:
        # make an entry in the dict for the customer
        topn[uid] = list()

        # iterate over all the articles and predict how likely the item would
        # be bought add if above a 95% threshold
        for aid in articles:
            est = model.predict(uid, aid).est
            if est < 0.95:
                continue

            topn[uid].append((aid, est))

        # filter out the top 12 items bought
        temp_topn = dict()
        for uid, est in topn.items():
            sorted_est = sorted(est, key=lambda x: x[1])
            temp_topn[uid] = sorted_est[:12]

        # save the top 12 items bought
        topn = temp_topn
        del temp_topn  # delete to save memory

    return topn


def get_top_product_type_interval(transactions, age_interval):
    """
    gets the top 12 product types for a given age interval
    :param transactions: the transactions to get the interval ranking from
    :param age_interval: a list of values in an interval
    :return: top 12 product types for the given interval
    """
    # take only transactions in cur_interval
    interval_transactions = transactions[transactions['age'].isin(age_interval)]

    # determine the top product types in the age interval
    ranking = interval_transactions \
        .groupby('product_type_no')['product_type_no'].sum() \
        .rank(method='dense').rename('rank') \
        .astype(np.uint8) \
        .reset_index() \
        .sort_values('rank').head(12)

    # convert this ranking to numpy to return it as a list
    return ranking.to_numpy()


def get_last_weeks_popularity(pop_type, transactions):
    """
    get the popularity of articles in raw numbers per week
    :param pop_type: the type of popularity we use, e.g. for articles ids or
    product types
    :param transactions: the transactions dataframe
    :return: transactions
    """
    if pop_type == 'item_popularity':
        transactions = get_last_weeks_popularity_el(transactions, 'article_id',
                                                    pop_type)
    elif pop_type == 'colour_popularity':
        transactions = get_last_weeks_popularity_el(transactions,
                                                    'colour_group_code',
                                                    pop_type)
    elif pop_type == 'product_type_popularity':
        transactions = get_last_weeks_popularity_el(transactions,
                                                    'product_type_no', pop_type)
    return transactions[pop_type]


def get_last_weeks_popularity_el(transactions, group_type, pop_type):
    """
    gets the popularity of items of the previous week
    :param transactions: transactions datframe
    :param group_type: group type to use for popularity
    :param pop_type: popularity type to use for setting the popularity column on
    whether an article has been ordered
    :return: transactions with previous weeks popularity
    """
    # copy transactions to change it
    transactions_up_shift_week = transactions.copy()
    transactions_up_shift_week['week'] += 1  # shift the week by 1

    # get the popularity of specified group type by week, coincidentally this is
    # also the previous weeks one
    pop = \
        transactions_up_shift_week.groupby(['week', group_type],
                                           as_index=False)[['ordered']].sum()

    # set type of pop type to ordered count
    pop[pop_type] = pop['ordered']

    # drop the ordered column
    pop = pop.drop('ordered', axis=1)

    # merge the popularity with the transactions so that it becomes the
    # previous weeks popularity in transactions itself
    transactions = transactions.merge(pop, on=['week', group_type])

    return transactions
