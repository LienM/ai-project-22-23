import pandas as pd
from collections import defaultdict
from utils import DATA_PATH, customer_hex_id_to_int

if __name__ == '__main__':
    # create a validation set, which is the last week of the transactions dataset
    # source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
    articles = pd.read_parquet(f'{DATA_PATH}/articles.parquet')
    customers = pd.read_parquet(f'{DATA_PATH}/customers.parquet')
    transactions = pd.read_parquet(f'{DATA_PATH}/transactions_train.parquet')

    val_week_purchases_by_cust = defaultdict(list)

    val_week_purchases_by_cust.update(
        transactions[transactions.week == transactions.week.max()]
        .groupby('customer_id')['article_id']
        .apply(list)
        .to_dict()
    )

    sample_sub = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
    valid_gt = customer_hex_id_to_int(sample_sub.customer_id) \
        .map(val_week_purchases_by_cust) \
        .apply(lambda xx: ' '.join('0' + str(x) for x in xx))

    sample_sub.prediction = valid_gt
    sample_sub.to_parquet(
        f'{DATA_PATH}/validation_ground_truth.parquet', index=False)