import pandas as pd
from recpack.algorithms import ItemKNN, Prod2Vec
from recpack.pipelines import PipelineBuilder
from recpack.preprocessing.filters import Deduplicate, MinUsersPerItem
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.scenarios import StrongGeneralizationTimed
from multiprocessing import Pool

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


def get_recpack_samples(transactions, n=12, algorithm='prod2vec', customers=None):
    """
    Get top n items based on a recpack algorithm, for each customer with a purchase history
    :param transactions:
    :param n:
    :return:
    """
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    interaction_matrix = df_pp.process(transactions)
    if algorithm == 'itemknn':
        algo = ItemKNN(K=100)
        algo.fit(interaction_matrix)
    elif algorithm == 'prod2vec':
        scenario = StrongGeneralizationTimed(0.75, validation=True, t=transactions['week'].max() + 1,
                                             t_validation=transactions['week'].max())
        scenario.split(interaction_matrix)
        algo = Prod2Vec()
        algo.fit(scenario.full_training_data, (scenario.validation_data_in, scenario.validation_data_out))
    else:
        raise ValueError("Algorithm not supported")

    predictions = algo.predict(interaction_matrix)
    valid_customers = customers
    customers = transactions['customer_id'].unique()
    articles = transactions['article_id'].unique()
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


def recpack_cv(data):
    """
    Function to compare some algorithms in recpack
    :param data:
    :return:
    """
    transactions = data['transactions']
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    interaction_matrix = df_pp.process(transactions)
    scenario = StrongGeneralizationTimed(0.75, validation=True, t=data['transactions']['week'].max(),
                                         t_validation=data['transactions']['week'].max() - 4)
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


def add_recpack_score(samples, transactions, algorithm='prod2vec'):
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    scores = pd.DataFrame()
    for w in tqdm(samples.week.unique(), desc="Adding recpack scores", leave=False):
        transactions_w = transactions[transactions.week < w]
        customer_map = {c: i for i, c in enumerate(transactions_w.customer_id.unique())}
        article_map = {a: i for i, a in enumerate(transactions_w.article_id.unique())}
        interaction_matrix = df_pp.process(transactions_w)
        if algorithm == 'itemknn':
            algo = ItemKNN(K=100)
            algo.fit(interaction_matrix)
        elif algorithm == 'prod2vec':
            scenario = StrongGeneralizationTimed(0.75, validation=True, t=transactions['week'].max() + 1,
                                                 t_validation=transactions['week'].max())
            scenario.split(interaction_matrix)
            algo = Prod2Vec()
            algo.fit(scenario.full_training_data, (scenario.validation_data_in, scenario.validation_data_out))
        else:
            raise ValueError("Algorithm not supported")

        predictions = algo.predict(interaction_matrix)
        scores_w = samples[samples.week == w][['customer_id', 'article_id', 'week']]
        scores_w[f'{algorithm}_score'] = scores_w. \
            progress_apply(
            lambda x: predictions[customer_map[x[0]], article_map[x[1]]] if x[0] in customer_map and x[
                1] in article_map.keys() else float(0),
            axis=1, raw=True)
        scores = pd.concat([scores, scores_w])
    samples = merge_downcast(samples, scores, on=['customer_id', 'article_id', 'week'], how='left')
    return samples


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    data = load_data(frac=0.05)
    data = pp_data(data, {'w2v_vector_size': 0}, force=True, write=False)
    transactions = data['transactions']
    data['transactions'] = transactions[transactions['week'] > transactions['week'].max() - 8]
    recpack_cv(data)
