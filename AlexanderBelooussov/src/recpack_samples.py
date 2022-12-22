import pandas as pd
from recpack.algorithms import ItemKNN, Prod2Vec
from recpack.pipelines import PipelineBuilder
from recpack.preprocessing.filters import Deduplicate, MinUsersPerItem
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.scenarios import StrongGeneralizationTimed

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None

from utils import *
from preprocessing import *


# def get_recpack_samples_for_customer(customer, i, predictions, articles, n=12):
#     customer_predictions = predictions.getrow(i)
#     articles_sorted = np.argsort(customer_predictions)
#     top_n = articles_sorted[:n]
#     top_n = [articles[x] for x in top_n]
#     df = pd.DataFrame({'customer_id': customer, 'article_id': top_n})
#     return df


def get_recpack_samples(transactions_train, n=12, algorithm='prod2vec', customers=None):
    """
    Get top n items based on a recpack algorithm, for each customer with a purchase history
    For week after last week in transactions
    :param transactions_train: DataFrame with transactions
    :param n: number of items to recommend
    :param algorithm: algorithm to use, ['itemknn', 'prod2vec']
    :param customers: list of customers to recommend for
    :return: DataFrame with samples
    """
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    interaction_matrix = df_pp.process(transactions_train)
    predictions = get_predictions(algorithm, interaction_matrix, transactions_train)
    valid_customers = customers
    customers = transactions_train['customer_id'].unique()
    articles = transactions_train['article_id'].unique()
    # get top 12 predictions for each customer
    results = []
    for i, customer in enumerate(tqdm(customers, desc=f"Generating top {n} recpack samples", leave=False)):
        if valid_customers is not None and customer not in valid_customers:
            continue
        customer_predictions = predictions.getrow(i).toarray()[0]
        articles_sorted = np.argsort(customer_predictions)
        top_n = articles_sorted[:n]
        top_n = [articles[x] for x in top_n]
        df = pd.DataFrame({'customer_id': customer, 'article_id': top_n})
        results.append(df)
    result_df = concat_downcast(results)
    return result_df.reset_index(drop=True)


def recpack_cv(data_dict):
    """
    Function to compare some algorithms in recpack
    :param data_dict: dictionary with data
    :return:
    """
    transactions_train = data_dict['transactions']
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    interaction_matrix = df_pp.process(transactions_train)
    scenario = StrongGeneralizationTimed(0.75, validation=True, t=data_dict['transactions']['week'].max(),
                                         t_validation=data_dict['transactions']['week'].max() - 4)
    scenario.split(interaction_matrix)
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)
    builder.add_algorithm('Popularity')  # No real parameters to optimise
    builder.add_algorithm('ItemKNN', grid={
        'K': [100, 500, 1000, 5000, 10000],
    })
    builder.add_algorithm('Prod2Vec', grid={
        # 'num_components': [25, 50],
        # 'K': [200, 500],
        # 'num_negatives': [1, 5, 10],
    })
    builder.set_optimisation_metric('NDCGK', K=12)
    builder.add_metric('NDCGK', K=[12, 50, 100, 300])
    builder.add_metric('CoverageK', K=[12, 50, 100, 300])
    pipeline = builder.build()
    pipeline.run()
    print(pd.DataFrame.from_dict(pipeline.get_metrics()))


def add_recpack_score(samples, transactions_train, algorithm='prod2vec'):
    """
    Add recpack score to samples, after all samples have been generated
    :param samples: DataFrame with samples
    :param transactions_train: DataFrame with transactions
    :param algorithm: algorithm to use, ['itemknn', 'prod2vec']
    :return: DataFrame with samples and scores
    """
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    scores = pd.DataFrame()
    for w in tqdm(samples.week.unique(), desc="Adding recpack scores", leave=False):
        transactions_w = transactions_train[transactions_train.week < w]
        customer_map = {c: i for i, c in enumerate(transactions_w.customer_id.unique())}
        article_map = {a: i for i, a in enumerate(transactions_w.article_id.unique())}
        interaction_matrix = df_pp.process(transactions_w)
        predictions = get_predictions(algorithm, interaction_matrix, transactions_train)
        scores_w = samples[samples.week == w][['customer_id', 'article_id', 'week']]
        scores_w[f'{algorithm}_score'] = scores_w. \
            progress_apply(
            lambda x: predictions[customer_map[x[0]], article_map[x[1]]] if x[0] in customer_map and x[
                1] in article_map.keys() else float(0),
            axis=1, raw=True)
        scores = pd.concat([scores, scores_w])
    samples = merge_downcast(samples, scores, on=['customer_id', 'article_id', 'week'], how='left')
    return samples


def get_predictions(algorithm, interaction_matrix, transactions_train):
    """
    Get predictions for a given algorithm
    :param algorithm: algorithm to use, ['itemknn', 'prod2vec']
    :param interaction_matrix: interaction matrix generated by Recpack
    :param transactions_train: DataFrame with transactions
    :return: Matrix with predictions
    """
    if algorithm == 'itemknn':
        algo = ItemKNN(K=100)
        algo.fit(interaction_matrix)
    elif algorithm == 'prod2vec':
        scenario = StrongGeneralizationTimed(0.75, validation=True, t=transactions_train['week'].max() + 1,
                                             t_validation=transactions_train['week'].max())
        scenario.split(interaction_matrix)
        algo = Prod2Vec()
        algo.fit(scenario.full_training_data, (scenario.validation_data_in, scenario.validation_data_out))
    else:
        raise ValueError("Algorithm not supported")
    predictions = algo.predict(interaction_matrix)
    return predictions


def generate_recpack_samples(n, n_train_weeks, previous_week_info, recpack_candidates, recpack_methods, scale_n,
                             test_set_transactions, transactions_train, unique_transactions, verbose, only_candidates):
    """
    Generate samples using recpack
    For all n_train_weeks and test week
    :param n: Int, number of samples to generate
    :param n_train_weeks: Int, number of weeks to use for training
    :param previous_week_info: DataFrame with previous week info
    :param recpack_candidates: DataFrame with recpack candidates
    :param recpack_methods: List with recpack methods
    :param scale_n: Bool, whether to scale up n as the week gets closer to the test week
    :param test_set_transactions: DataFrame, customers for test week
    :param transactions_train: DataFrame with transactions
    :param unique_transactions: DataFrame, unique customer interactions (per week)
    :param verbose: Bool, whether to print progress
    :param only_candidates: Bool, whether to only generate candidates
    :return: DataFrame with samples
    """
    for method in recpack_methods:
        if verbose:
            print(f"Generating samples with RecPack: {method}")
        # some initial recpack candidates
        start = transactions_train.week.max() - n_train_weeks + 1 if not only_candidates \
            else transactions_train.week.max() + 1
        for w in tqdm(range(start, transactions_train.week.max() + 2),
                      desc=f"Recpack samples per week"):
            transactions_w = transactions_train[transactions_train.week < w]
            if scale_n:
                i = int((w - transactions_train.week.min()) / (
                        transactions_train.week.max() + 1 - transactions_train.week.min()) * n)
                i = max(1, i)
            else:
                i = n
            if w < transactions_train.week.max() + 1:
                customers_to_use = unique_transactions[unique_transactions.week == w]
                customers_to_use = customers_to_use[customers_to_use.customer_id.isin(transactions_w.customer_id)]
            else:
                customers_to_use = test_set_transactions

            recpack = get_recpack_samples(transactions_w, i, algorithm=method,
                                          customers=customers_to_use.customer_id.unique())
            recpack['week'] = w
            recpack_samples = pd.merge(
                customers_to_use,
                recpack,
                on=['week', 'customer_id']
            )
            if w < transactions_train.week.max() + 1:
                assert recpack_samples.shape[0] == customers_to_use.shape[0] * i, \
                    f"Expected {customers_to_use.shape[0]} * {i} = {customers_to_use.shape[0] * i} samples, " \
                    f"got {recpack_samples.shape[0]}"
            else:
                assert recpack_samples.shape[0] == customers_to_use.shape[0] * i, \
                    f"{recpack_samples.shape[0]} != {customers_to_use.shape[0]} * {i}"
            recpack_candidates = concat_downcast([recpack_candidates, recpack_samples])
            recpack_candidates.drop_duplicates(['week', 'customer_id', 'article_id'], inplace=True)
    # add price
    if len(recpack_methods):
        recpack_candidates = merge_downcast(recpack_candidates, previous_week_info[['week', 'article_id', 'price']],
                                            on=['week', 'article_id'], how='left')
        recpack_candidates.drop_duplicates(['week', 'customer_id', 'article_id'], inplace=True)
    return recpack_candidates


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    data = load_data(frac=0.05)
    data = pp_data(data, {'w2v_vector_size': 0}, force=True, write=False)
    transactions = data['transactions']
    data['transactions'] = transactions[transactions['week'] > transactions['week'].max() - 8]
    recpack_cv(data)
