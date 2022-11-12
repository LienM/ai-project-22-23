from recpack.scenarios import StrongGeneralizationTimed
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import Deduplicate, MinUsersPerItem
from recpack.pipelines import PipelineBuilder
from recpack.algorithms import STAN
import pandas as pd
import numpy as np
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None


def recpack_samples(transactions, n=12):
    df_pp = DataFramePreprocessor(user_ix='customer_id', item_ix='article_id', timestamp_ix='week')
    df_pp.add_filter(
        Deduplicate("article_id", "customer_id", "week")
    )
    df_pp.add_filter(MinUsersPerItem(1, "article_id", "customer_id"))
    interaction_matrix = df_pp.process(transactions)

    algo = STAN()
    algo.fit(interaction_matrix)
    predictions = algo.predict(interaction_matrix).toarray()
    customers = transactions['customer_id'].unique()
    articles = transactions['article_id'].unique()
    # get top 12 predictions for each customer
    result_df = pd.DataFrame()
    for i, customer in enumerate(customers):
        customer_predictions = predictions[i]
        articles_sorted = np.argsort(customer_predictions)
        top_n = articles_sorted[:n]
        top_n = [articles[x] for x in top_n]
        df = pd.DataFrame({'customer_id': customer, 'article_id': top_n})
        result_df = result_df.append(df)

    return result_df.reset_index(drop=True)


# scenario = StrongGeneralizationTimed(0.75, validation=True, t=data['transactions']['week'].max(), t_validation=data['transactions']['week'].max()-1)
    # scenario.split(interaction_matrix)
    # builder = PipelineBuilder()
    # builder.set_data_from_scenario(scenario)
    # builder.add_algorithm('Popularity')  # No real parameters to optimise
    # builder.add_algorithm('BPRMF', grid={
    #     'num_components': [10, 20, 50, 100],
    # })
    # builder.add_algorithm('Prod2Vec', grid={
    #     # 'K': [25, 50, 100],
    # })
    # builder.add_algorithm('STAN')
    # builder.add_algorithm('SLIM')
    # builder.add_algorithm('TARSItemKNN')
    # builder.set_optimisation_metric('NDCGK', K=12)
    # builder.add_metric('NDCGK', K=[12])
    # builder.add_metric('CoverageK', K=[12])
    # pipeline = builder.build()
    # pipeline.run()
    # print(pd.DataFrame.from_dict(pipeline.get_metrics()))
