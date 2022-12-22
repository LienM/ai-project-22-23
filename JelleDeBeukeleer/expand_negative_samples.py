import pandas as pd
import json

settings_file = "./settings.json"
settings = json.load(open(settings_file))
data_dir = settings["data_directory"]
filenames = settings["data_filenames"]["processed"]
articles_fn = data_dir + filenames["articles"]
customers_fn = data_dir + filenames["customers"]
transactions_fn = data_dir + filenames["transactions"]
bestsellers_fn = data_dir + filenames["bestsellers"]
customer_df: pd.DataFrame = pd.read_csv(customers_fn)
articles_df = pd.read_csv(articles_fn)
bestsellers = pd.read_csv(bestsellers_fn)
magic_number = 5


def append_negatives(df: pd.DataFrame, neg_samples: pd.DataFrame, week: int):
    global customer_df
    global articles_df
    global bestsellers

    neg_samples["ordered"] = 0
    neg_samples["price_discrepancy"] = neg_samples["average_article_price"] - \
                                       neg_samples["average_customer_budget"]
    neg_samples["t_dat"] = week
    neg_samples["price"] = neg_samples["average_article_price"]
    neg_samples["sales_channel_id"] = 2
    neg_samples = neg_samples.merge(bestsellers, how="left", on=["t_dat", "article_id"])
    neg_samples["bestseller_rank"].fillna(999, inplace=True)
    neg_samples = neg_samples.reindex(columns=df.columns)

    df = pd.concat([df, neg_samples])
    df.drop_duplicates(subset=["customer_id", "article_id", "t_dat"], inplace=True)
    return df


def random_negatives(df: pd.DataFrame, week: int = None):
    global magic_number
    if not week:
        week = df["t_dat"].max()
    article_sample_count = int(len(df[df.t_dat==week]) / magic_number)
    neg_samples = customer_df.sample(n=magic_number).copy()
    neg_samples = neg_samples.merge(
        articles_df.sample(n=article_sample_count).copy(), how="cross")


    return append_negatives(df, neg_samples, week)


def focused_negatives(df: pd.DataFrame, week: int):
    positive_view = df.ordered == 1
    neg_samples = df[positive_view].copy()
    original_week = week + 1
    neg_samples = neg_samples[neg_samples["t_dat"] == original_week].sample(frac=0.01)
    neg_samples.drop("bestseller_rank", inplace=True, axis=1)
    return append_negatives(df, neg_samples, week)


if __name__ == "__main__":
    transactions = pd.read_csv(transactions_fn)
    for i in transactions.t_dat.unique():
        transactions = focused_negatives(transactions, i)
        transactions = random_negatives(transactions, i)

    print(transactions.ordered.value_counts())
    transactions.to_csv(transactions_fn, index=False)

