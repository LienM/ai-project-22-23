import pandas as pd
from utils import DATA_PATH

if __name__ == '__main__':
    articles = pd.read_parquet(f'{DATA_PATH}/articles.parquet')
    customers = pd.read_parquet(f'{DATA_PATH}/customers.parquet')
    transactions = pd.read_parquet(f'{DATA_PATH}/transactions_train.parquet')

    # generate a 5% sample of the complete dataset
    # source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
    sample = 0.05
    customers_sample = customers.sample(frac=sample, replace=False)
    customers_sample_ids = set(customers_sample['customer_id'])
    transactions_sample = transactions[transactions["customer_id"].isin(customers_sample_ids)]
    articles_sample_ids = set(transactions_sample["article_id"])
    articles_sample = articles[articles["article_id"].isin(articles_sample_ids)]

    customers_sample.to_parquet(f'{DATA_PATH}/customers_sample_{sample}.parquet', index=False)
    transactions_sample.to_parquet(f'{DATA_PATH}/transactions_train_sample_{sample}.parquet', index=False)
    articles_sample.to_parquet(f'{DATA_PATH}/articles_train_sample_{sample}.parquet', index=False)