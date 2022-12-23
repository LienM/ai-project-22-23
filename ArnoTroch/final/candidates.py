# contains all different candidate generation methods used across the notebooks
import numpy as np
import pandas as pd
from recpack.algorithms import TARSItemKNN
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.filters import MinUsersPerItem, MinItemsPerUser
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from scipy.sparse import csr_matrix


############################################################################################################
# Helper functions
############################################################################################################

def top_n_idx_sparse(matrix: csr_matrix, n: int):
    """
    Return index of top n values in each row of a sparse matrix.
    source: https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

    :param matrix: sparse matrix
    :param n: number of top values to return
    :return: two lists: one with the indices of the top n values, one with the values themselves
    """
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]].tolist())

    # Get the values corresponding to the indices
    top_n_values = []
    for row_idx, col_idxs in enumerate(top_n_idx):
        top_n_values.append(matrix[row_idx, col_idxs].toarray().tolist()[0])
        assert (len(top_n_values[row_idx]) == len(top_n_idx[row_idx]))
    return top_n_idx, top_n_values


def get_top_k_similar_articles_per_user(
        prediction_matrix: csr_matrix,
        interaction_matrix: InteractionMatrix,
        k: int) -> pd.DataFrame:
    """
    given a prediction matrix and a transaction matrix, return a dataframe with the top k similar articles per user.

    :param prediction_matrix: prediction matrix
    :param interaction_matrix: transaction matrix
    :param k: number of similar articles to return
    """
    # use interaction_matrix._df to map back to original customer and article ids
    uid_cid_map = interaction_matrix._df[["uid", "customer_id"]].drop_duplicates() \
        .set_index("uid").to_dict()["customer_id"]
    iid_aid_map = interaction_matrix._df[["iid", "article_id"]].drop_duplicates() \
        .set_index("iid").to_dict()["article_id"]

    # get column indices of top k articles per user
    top_k_idx, top_k_values = top_n_idx_sparse(prediction_matrix, k)

    similar_customers = []
    similar_articles = []
    similarity_scores = []

    for i, row in enumerate(top_k_idx):
        user_predictions = [iid_aid_map[iid] for iid in row]
        similar_customers.extend([uid_cid_map[i]] * len(user_predictions))
        similar_articles.extend(user_predictions)
        similarity_scores.extend(top_k_values[i])

    assert len(similar_customers) == len(similar_articles) == len(similarity_scores), "lengths of lists should be equal"
    return pd.DataFrame({
        "customer_id": similar_customers,
        "article_id": similar_articles,
        "similarity": similarity_scores
    })


def calculate_popularity(row, popularity, weekly_popularity):
    """
    helper function to calculate popularity of an article for a single row in a dataframe.
    :param row: row in a dataframe
    :param popularity: the dataframe where the count of each article is stored per week
    :param weekly_popularity: list with weekly popularity of each article
    :return: time_weighted weekly popularity score
    """
    weeks_before = popularity[(row.article_id == popularity.article_id) & (row.week > popularity.week)]
    # get last row of weeks_before
    previous_week_popularity = 0
    if weeks_before.shape[0] > 0:
        previous_week_popularity = weekly_popularity[-1]
    return previous_week_popularity / 2.0 + float(row.weekly_purchase_count)


############################################################################################################
# Candidate generation functions
############################################################################################################

def generate_last_purchased_candidates(df, test_week):
    """
    For each user, return the last purchased items as candidates.
    Source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03a_Basic_Model_Local_Validation.ipynb

    :param df: transactions dataframe
    :param test_week: week to predict
    :return: dataframe with same column names as df
    """
    # for each customer, get all the weeks where they have made purchases.
    # then, for each of those weeks, map to the previous week where the customer made a purchase
    c2weeks = df.groupby('customer_id')['week'].unique()
    c2weeks2shifted_weeks = {}

    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

    # generate candidates
    candidates_last_purchase = df.copy()
    weeks = []
    for i, (c_id, week) in enumerate(zip(df['customer_id'], df['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])
    candidates_last_purchase.week = weeks

    return candidates_last_purchase


def generate_bestseller_candidates(df, test_week, n=12):
    """
    For each user and week, return the n bestselling items from last week as candidates for the user.
    Source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/03a_Basic_Model_Local_Validation.ipynb

    :param df: transactions dataframe
    :param test_week: week to predict
    :param n: number of candidates to generate
    :return: dataframe with candidates, as well as the n bestselling items from each previous week
    """
    # first, find the bestselling items for each week
    mean_price = df.groupby(['week', 'article_id'])['price'].mean()
    sales = df \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='dense', ascending=False) \
        .groupby('week').head(n).rename('bestseller_rank').astype('int8')

    bestsellers_previous_week = pd.merge(sales, mean_price, on=['week', 'article_id']).reset_index()
    bestsellers_previous_week.week += 1

    # for each week, get all unique customers who bought something
    unique_customers = df.groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy()

    # generate candidates for those customers
    candidates_bestsellers = pd.merge(
        unique_customers,
        bestsellers_previous_week,
        on='week',
    )

    # generate candidates for the test week for all users who have bought something
    test_set_df = unique_customers.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_df.week = test_week
    candidates_bestsellers_test_week = pd.merge(
        test_set_df,
        bestsellers_previous_week,
        on='week'
    )

    # combine the two dataframes
    candidates_bestsellers = pd.concat([candidates_bestsellers, candidates_bestsellers_test_week])
    candidates_bestsellers.drop(columns='bestseller_rank', inplace=True)
    return candidates_bestsellers, bestsellers_previous_week


def generate_similar_candidates(df, test_week, n=12):
    """
    For each user, return the n most similar items to the user's purchased items as candidates.

    :param df: transactions dataframe
    :param test_week: week to predict
    :param n: number of candidates to generate
    :return: dataframe with candidates, as well as the n most similar items to each user's purchased items
    """
    # preprocessing
    proc = DataFramePreprocessor(item_ix='article_id', user_ix='customer_id', timestamp_ix='week')
    proc.add_filter(MinUsersPerItem(10, item_ix='article_id', user_ix='customer_id'))
    proc.add_filter(MinItemsPerUser(10, item_ix='article_id', user_ix='customer_id'))
    transaction_matrix = proc.process(df)

    # train model
    knn = TARSItemKNN(K=580, fit_decay=0.1, predict_decay=1 / 3, similarity='cosine')
    knn.fit(transaction_matrix)

    # for each user, get the top n most similar items to the user's purchased items
    prediction_matrix = knn.predict(transaction_matrix)
    similar_items = get_top_k_similar_articles_per_user(prediction_matrix, transaction_matrix, k=n)

    # get all unique customers who bought something, and generate candidates for them in the test week
    test_set_df = df \
        .groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy() \
        .drop_duplicates('customer_id').reset_index(drop=True)
    test_set_df.week = test_week

    candidates_similar_items = pd.merge(
        similar_items,
        test_set_df,
        on='customer_id',
        how='left'
    )
    candidates_similar_items.drop(columns='similarity', inplace=True)

    # get the last price of all article_id's and add them to the candidates
    last_price = df \
        .groupby(['article_id', 'week']).price.last().reset_index() \
        .groupby('article_id').price.last().reset_index()

    candidates_similar_items = pd.merge(
        candidates_similar_items,
        last_price,
        on='article_id',
        how='left'
    )
    return candidates_similar_items, similar_items


def generate_popularity_candidates(df, test_week, n=12):
    """
    For each user and week, return the n most popular items from last week as candidates for the user.

    :param df: transactions dataframe
    :param test_week: week to predict
    :param n: number of candidates to generate
    :return: dataframe with candidates, as well as the n most popular items from each previous week
    """
    # calculate the popularity of each item (i.e., the number of times it was bought in each week)
    popularity = df.groupby(['article_id', 'week']).size().reset_index(name='weekly_purchase_count')

    # iterate over all rows and calculate weekly popularity (yes, that's very inefficient)
    weekly_popularity = []
    for i, row in popularity.iterrows():
        weekly_popularity.append(calculate_popularity(row, popularity, weekly_popularity))
    popularity['weekly_popularity'] = weekly_popularity

    # keep only the top n most popular items for each week
    popular_articles_per_week = popularity.sort_values(['week', 'weekly_popularity'], ascending=False) \
        .groupby('week').head(n).reset_index(drop=True)

    # add prices to the articles
    mean_price = df.groupby(['week', 'article_id'])['price'].mean()
    popular_articles_previous_week = pd.merge(
        popular_articles_per_week,
        mean_price,
        on=['week', 'article_id']
    ).reset_index(drop=True)

    # make a new column to rank the weekly_popularity
    popular_articles_previous_week['last_week_popularity_rank'] = popular_articles_previous_week \
        .groupby('week')['weekly_popularity'].rank(ascending=False).astype(np.int32)
    popular_articles_previous_week.week += 1

    # for each week, get all unique customers who bought something
    unique_customers = df.groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy()

    # generate candidates for those customers
    candidates_most_popular = pd.merge(
        unique_customers,
        popular_articles_previous_week,
        on='week',
    )

    # generate candidates for the test week for all users who have bought something
    test_set_df = unique_customers.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_df.week = test_week
    candidates_most_popular_test_week = pd.merge(
        test_set_df,
        popular_articles_previous_week,
        on='week'
    )

    # combine the two dataframes
    candidates_most_popular = pd.concat([candidates_most_popular, candidates_most_popular_test_week])
    candidates_most_popular.drop(
        columns=['weekly_purchase_count', 'weekly_popularity', 'last_week_popularity_rank'],
        inplace=True
    )
    return candidates_most_popular, popular_articles_previous_week
