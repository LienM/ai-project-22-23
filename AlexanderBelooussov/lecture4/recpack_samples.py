import pandas as pd
from recpack.algorithms import ItemKNN
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


def get_recpack_samples(transactions, n=12):
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
    algo = ItemKNN(K=100)
    algo.fit(interaction_matrix)
    predictions = algo.predict(interaction_matrix)
    customers = transactions['customer_id'].unique()
    articles = transactions['article_id'].unique()
    # get top 12 predictions for each customer
    results = []
    for i, customer in enumerate(tqdm(customers, desc=f"Generating top {n} recpack samples", leave=False)):
        customer_predictions = predictions.getrow(i)
        articles_sorted = np.argsort(customer_predictions)
        top_n = articles_sorted[:n]
        top_n = [articles[x] for x in top_n]
        df = pd.DataFrame({'customer_id': customer, 'article_id': top_n})
        results.append(df)

    # repeat using multiprocessing Pool
    # pool = Pool(6)
    # for r in tqdm(pool.starmap(get_recpack_samples_for_customer,
    #                             [(customer, i, predictions, articles, n) for i, customer in enumerate(customers)])
    #               , desc=f"Generating top {n} recpack samples", total=len(customers), leave=False):
    #     results.append(r)
    # with Pool(7) as p:
    #     results = p.starmap(get_recpack_samples_for_customer,
    #                         [(customer, i, predictions, articles, n) for i, customer in enumerate(customers)])

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
    builder.set_optimisation_metric('NDCGK', K=12)
    builder.add_metric('NDCGK', K=[12, 50, 100, 300])
    builder.add_metric('CoverageK', K=[12, 50, 100, 300])
    pipeline = builder.build()
    pipeline.run()
    print(pd.DataFrame.from_dict(pipeline.get_metrics()))


def add_recpack_score(samples, transactions):
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    for w in tqdm(transactions.week.unique()[1:], desc="Adding recpack scores", leave=False):
        transactions_w = transactions[transactions.week < w]
        customer_map = {c: i for i, c in enumerate(transactions_w.customer_id.unique())}
        article_map = {a: i for i, a in enumerate(transactions_w.article_id.unique())}
        interaction_matrix = df_pp.process(transactions_w)
        algo = ItemKNN(K=100)
        algo.fit(interaction_matrix)
        predictions = algo.predict(interaction_matrix)

        samples['recpack_score'] = samples.loc[samples.week == w][['customer_id', 'article_id']]. \
            progress_apply(
            lambda x: predictions[customer_map[x[0]], article_map[x[1]]] if x[0] in customer_map and x[
                1] in article_map else float(0),
            axis=1, raw=True)
        break
    return samples


if __name__ == '__main__':
    data = load_data(frac=1)
    data = pp_data(data, {'w2v_vector_size': 0}, force=True, write=False)
    transactions = data['transactions']
    data['transactions'] = transactions[transactions['week'] > transactions['week'].max() - 12]
    recpack_cv(data)
