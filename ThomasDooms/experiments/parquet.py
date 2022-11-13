import os.path
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def customer_id_to_int(x): return int(x[-16:], 16)


def benchmark_read():
    formats = {
        # 'csv': (pd.read_csv, 'to_csv'),
        'parquet': (pd.read_parquet, 'to_parquet'),
        'feather': (pd.read_feather, 'to_feather'),
    }

    base = '../data/transactions'
    initial = pd.read_feather(f'{base}.feather')

    for fmt, (reader, writer) in formats.items():
        path = f'{base}.{fmt}'

        if not os.path.exists(path):
            getattr(initial, writer)(path)

        start = time.time()
        reader(path)
        print(f'{fmt}: {time.time() - start:.2f}s')


def simplify_transactions():
    start = time.time()
    transactions = pd.read_csv('../data/transactions.csv')
    # transactions = pd.read_feather('../data/transactions.feather')
    print("reading transactions:", time.time() - start)

    start = time.time()
    transactions['customer_id'] = transactions['customer_id'].apply(customer_id_to_int).astype('int32')
    transactions['article_id'] = transactions['article_id'].astype('int32')

    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d')
    # transactions['year'] = (transactions['t_dat'].dt.year - 2000).astype('int8')
    # transactions['month'] = transactions['t_dat'].dt.month.astype('int8')
    # transactions['day'] = transactions['t_dat'].dt.day.astype('int8')

    # https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
    transactions['week'] = (104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7).astype('int8')
    transactions.drop('t_dat', axis=1, inplace=True)

    transactions['price'] = transactions['price'].astype('float32')
    transactions['sales_channel_id'] = transactions['sales_channel_id'].astype('int8')
    print("simplifying transactions:", time.time() - start)

    transactions.to_feather('../data/transactions.feather')


def simplify_customers():
    start = time.time()
    customers = pd.read_csv('../data/customers.csv')
    customers.info(memory_usage='deep')

    customers.fillna({"FN": 0, "Active": 0}, inplace=True)

    customers["FN"] = customers["FN"].astype('int8')
    customers["Active"] = customers["Active"].astype('int8')
    customers["club_member_status"] = pd.factorize(customers["club_member_status"])[0].astype('int8')
    customers["fashion_news_frequency"] = pd.factorize(customers["fashion_news_frequency"])[0].astype('int8')
    customers['customer_id'] = customers['customer_id'].apply(customer_id_to_int).astype('int64')
    customers['postal_code'] = pd.factorize(customers['postal_code'])[0].astype('int32')

    customers.info(memory_usage='deep')

    print("elapsed:", time.time() - start)

    print(customers)
    customers.to_feather('../data/customers.feather')


def simplify_articles(bert=True):
    articles = pd.read_csv('../data/articles.csv')
    articles.info(memory_usage='deep')

    articles['article_id'] = articles['article_id'].astype('int32')
    articles['graphical_appearance_no'] = pd.factorize(articles['graphical_appearance_no'])[0].astype('int8')
    articles['colour_group_code'] = articles['colour_group_code'].astype('int8')
    articles['perceived_colour_value_id'] = articles['perceived_colour_value_id'].astype('int8')
    articles['perceived_colour_master_id'] = articles['perceived_colour_master_id'].astype('int8')
    articles['department_no'] = articles['department_no'].astype('int16')
    articles['index_code'] = pd.factorize(articles['department_no'])[0].astype('int8')
    articles['index_group_no'] = articles['index_group_no'].astype('int8')
    articles['section_no'] = articles['section_no'].astype('int8')
    articles['garment_group_no'] = articles['garment_group_no'].astype('int16')

    if bert:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        features = model.encode(articles['prod_name'].values.tolist()).tolist()
        transformed = PCA(n_components=16).fit_transform(features).tolist()
        articles[[f'prod_name_{i}' for i in range(16)]] = pd.DataFrame(transformed)

        # of course this column has strings, very cool!
        features = model.encode(articles['detail_desc'].map(str).values.tolist()).tolist()
        transformed = PCA(n_components=16).fit_transform(features).tolist()
        articles[[f'detail_desc_{i}' for i in range(16)]] = pd.DataFrame(transformed)

        # aggregated = articles.apply(lambda x: f"{x['department_name']} {x['colour_group_name']}", axis=1)
        # features = model.encode(aggregated.values.tolist()).tolist()
        # transformed = PCA(n_components=8).fit_transform(features).tolist()
        # articles[[f'dep_colour_{i}' for i in range(8)]] = pd.DataFrame(transformed)

        articles.drop(['prod_name', 'detail_desc', 'product_type_name', 'detail_desc'], axis=1, inplace=True)

    articles.drop(["graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
                   "perceived_colour_master_name", "department_name", "index_name", "index_group_name", "section_name",
                   "garment_group_name"], axis=1, inplace=True)

    articles.info(memory_usage='deep')
    print(articles)
    articles.to_feather('../data/articles.feather')


def simplify_submission():
    submission = pd.read_csv('../data/example.csv')
    submission.to_feather('../data/example.feather')


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.width = None

    simplify_articles()
    # simplify_customers()
    # simplify_transactions()
    # simplify_submission()

    # pd.read_csv('data/customers.csv')
    # benchmark_read()
