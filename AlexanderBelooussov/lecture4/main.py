from candidates import *
from ranking import *
from preprocessing import *
from samples import *

if __name__ == '__main__':
    # load all customers
    _, customers, _ = load_data(read_articles=False, read_customers=True, read_transactions=False, frac=1)
    c_frames = []
    k = 8
    # divide into k groups
    dfs = np.array_split(customers, k)

    for i, customers in enumerate(tqdm(dfs, desc="RUNNING MAIN", leave=True)):
        # articles, customers, transactions = load_data(read_articles=not os.path.exists('pickles/articles.pkl'),
        #                                               read_customers=not os.path.exists('pickles/customers.pkl'),
        #                                               read_transactions=not os.path.exists('pickles/transactions.pkl'),
        #                                               frac=1.0/8.0)
        # frac=0.001)

        customers.reset_index(drop=True, inplace=True)
        articles, _, transactions = load_data(read_customers=False)
        # filter transactions and articles
        transactions = transactions[transactions['customer_id'].isin(customers['customer_id'])]
        transactions.reset_index(drop=True, inplace=True)
        articles = articles[articles['article_id'].isin(transactions['article_id'])]
        articles.reset_index(drop=True, inplace=True)

        articles, customers, transactions, cus_keys = pp_data(articles, customers, transactions, force=True)

        transactions, transactions_val = test_train_split(transactions)

        samples = generate_samples(articles, customers, transactions, force=True)

        # print(f"transactions Memory usage: {round(transactions.memory_usage().sum() / 1024 ** 2, 2)} MB")
        # print(f"transactions_val Memory usage: {round(transactions_val.memory_usage().sum() / 1024 ** 2, 2)} MB")
        # print(f"customers Memory usage: {round(customers.memory_usage().sum() / 1024 ** 2, 2)} MB")
        # print(f"articles Memory usage: {round(articles.memory_usage().sum() / 1024 ** 2, 2)} MB")
        # print(f"cus_keys Memory usage: {round(cus_keys.memory_usage().sum() / 1024 ** 2, 2)} MB")
        # print(f"samples Memory usage: {round(samples.memory_usage().sum() / 1024 ** 2, 2)} MB")

        candidates = generate_candidates(
            articles,
            customers,
            transactions,
            method='popular',
            period=7,
            k=300,
        )
        transactions.drop(columns=['t_dat'], inplace=True)

        predictions = {}
        model = lgbm_ranker_train(samples)
        for key in tqdm(candidates, desc='Predicting'):
            prediction = lgbm_ranker_predict(
                model,
                get_data_from_canditates(candidates[key], key, articles, customers, transactions)
            )
            predictions[key] = prediction
        predictions = dict_to_df(predictions)
        columns_titles = ["customer_id", "prediction"]
        predictions = predictions.reindex(columns=columns_titles)

        print(predictions.head())
        print(transactions_val.head())

        print(f"MAP@12 score: {map_at_12(predictions, transactions_val)}")

        # replace customer id again
        predictions.rename(columns={'customer_id': 'transformed'}, inplace=True)
        predictions = predictions.merge(cus_keys, how='left', on='transformed')
        predictions.drop(columns=['transformed'], inplace=True)

        write_submission(predictions, append=True if i == 0 else False)
