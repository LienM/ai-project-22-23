import os

import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import swifter


def w2v_articles(articles, verbose=True, vec_size=25):
    """
    Creates word2vec embeddings for the prod_name column
    :param articles:
    :param verbose:
    :param vec_size:
    :return:
    """
    if vec_size < 1:
        return articles

    if verbose:
        print(f"Generating word2vec encoding for articles", end="")

    # setup training set for word2vec
    train_frame = articles[
        ['article_id', 'product_code', 'prod_name', 'product_type_name', 'product_group_name',
         'graphical_appearance_name', 'department_name', 'index_name', 'index_group_name', 'section_name',
         'garment_group_name']].drop_duplicates()
    train_frame = train_frame.apply(lambda x: ','.join(x.astype(str)), axis=1)
    train_frame = pd.DataFrame({'clean': train_frame})
    data = [row.split(',') for row in train_frame['clean']]

    # initialise and train model
    model = Word2Vec(min_count=1,
                     vector_size=vec_size,
                     workers=7,
                     window=3,
                     sg=0)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=30)

    # convert all names into vectors
    articles["w2v"] = articles['article_id'].apply(lambda x: model.wv[str(x)])
    articles[[f"w2v_{i}" for i in range(vec_size)]] = pd.DataFrame(articles['w2v'].tolist(), index=articles.index)
    articles.drop(columns=['w2v'], inplace=True)

    if verbose:
        print(f"\rPreprocessing data... ", end="")
    return articles


def pp_articles(articles, transactions, params, verbose=True):
    """
    Preprocesses the articles data
    :param articles:
    :param transactions:
    :param params:
    :param verbose:
    :return:
    """
    # reduce memory usage
    articles['article_id'] = pd.to_numeric(articles['article_id'], downcast='integer')
    articles['product_code'] = pd.to_numeric(articles['product_code'], downcast='integer')
    articles['product_type_no'] = pd.to_numeric(articles['product_type_no'], downcast='integer')
    articles['colour_group_code'] = pd.to_numeric(articles['colour_group_code'], downcast='integer')
    articles['perceived_colour_value_id'] = pd.to_numeric(articles['perceived_colour_value_id'], downcast='integer')
    articles['perceived_colour_master_id'] = pd.to_numeric(articles['perceived_colour_master_id'], downcast='integer')
    articles['index_group_no'] = pd.to_numeric(articles['index_group_no'], downcast='integer')
    articles['section_no'] = pd.to_numeric(articles['section_no'], downcast='integer')

    articles = w2v_articles(articles, verbose, params['w2v_vector_size'])

    pn_encoder = LabelEncoder()
    articles['prod_name'] = pn_encoder.fit_transform(articles['prod_name'])

    articles.drop(
        columns=['product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                 'perceived_colour_master_name', 'department_name', 'index_name', 'index_group_name', 'section_name',
                 'garment_group_name', 'detail_desc'], inplace=True)

    gan_encoder = LabelEncoder()
    articles['graphical_appearance_no'] = gan_encoder.fit_transform(articles['graphical_appearance_no']).astype('int8')

    pgn_encoder = LabelEncoder()
    articles['product_group_name'] = pd.to_numeric(pgn_encoder.fit_transform(articles['product_group_name']),
                                                   downcast='integer')

    ic_encoder = LabelEncoder()
    articles['index_code'] = pd.to_numeric(ic_encoder.fit_transform(articles['index_code']), downcast='integer')

    dn_encoder = LabelEncoder()
    articles['department_no'] = dn_encoder.fit_transform(articles['department_no']).astype('int16')

    ggn_encoder = LabelEncoder()
    articles['garment_group_no'] = ggn_encoder.fit_transform(articles['garment_group_no']).astype('int8')

    # print(articles.head(20))
    # print(articles.info())
    # print(articles.describe())
    return articles


def customer_id_transform(df, target_row):
    """
    Helper function that transforms the customer_id column into integers
    :param df:
    :param target_row: Name of the target row, string
    :return: Modified dataframe
    """
    df[target_row] = df['customer_id'].swifter.progress_bar(False).apply(lambda x: int(x[-8:], 16)).astype('int32')
    return df


def pp_customers(customers, transactions):
    """
    Preprocesses the customers data
    :param customers:
    :param transactions:
    :return:
    """
    cus_keys = customers.copy()
    cus_keys.drop(columns=[x for x in customers.columns if x != 'customer_id'], inplace=True)
    cus_keys = customer_id_transform(cus_keys, 'transformed')
    customers['customer_id'] = cus_keys['transformed']
    customers['FN'] = customers['FN'].fillna(0)
    customers['Active'] = customers['Active'].fillna(0)
    customers['club_member_status'] = customers['club_member_status'].fillna('NA')
    cms_encoder = LabelEncoder()
    customers['club_member_status'] = cms_encoder.fit_transform(customers['club_member_status'])

    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NA')
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('NONE', 'None')
    fnf_encoder = LabelEncoder()
    customers['fashion_news_frequency'] = fnf_encoder.fit_transform(customers['fashion_news_frequency'])

    pc_encoder = LabelEncoder()
    customers['postal_code'] = pd.to_numeric(pc_encoder.fit_transform(customers['postal_code']), downcast='integer')

    # add age is null column
    customers['age_is_null'] = customers['age'].isnull().astype(int)
    # fill age == na with median value
    # is this needed?
    customers['age'] = customers['age'].fillna(customers['age'].median())

    # reduce memory usage
    customers['FN'] = pd.to_numeric(customers['FN'], downcast='integer')
    customers['Active'] = pd.to_numeric(customers['Active'], downcast='integer')
    customers['age'] = pd.to_numeric(customers['age'], downcast='integer')
    customers['club_member_status'] = pd.to_numeric(customers['club_member_status'], downcast='integer')
    customers['fashion_news_frequency'] = pd.to_numeric(customers['fashion_news_frequency'], downcast='integer')
    customers['age_is_null'] = pd.to_numeric(customers['age_is_null'], downcast='integer')

    # print(customers.head(20))
    # print(customers.info())
    # print(customers.describe())
    return customers, cus_keys


def pp_transactions(transactions):
    """
    Preprocesses the transactions data
    :param transactions:
    :return:
    """
    # reduce memory usage
    transactions['article_id'] = pd.to_numeric(transactions['article_id'], downcast='integer')
    transactions['sales_channel_id'] = pd.to_numeric(transactions['sales_channel_id'], downcast='integer')
    transactions['customer_id'] = transactions['customer_id'].apply(lambda x: int(x[-8:], 16)).astype('int32')
    # print(transactions.info())

    # add week no from start of data
    start = transactions['t_dat'].min() - pd.Timedelta(
        days=1)  # week starts on wednesday, but first day in data is thursday
    # transactions['week'] = transactions['t_dat'].swifter.progress_bar(enable=True, desc="Adding week column").apply(
    #     lambda x: (x - start).days // 7).astype('int16')

    transactions['week'] = ((transactions.t_dat - start).dt.days // 7).astype('int16')

    # add day of week
    # transactions['day_of_week'] = (transactions.t_dat.dt.dayofweek).astype('int8')
    # dow_encoder = LabelEncoder()
    # transactions['day_of_week'] = pd.to_numeric(dow_encoder.fit_transform(transactions['day_of_week']),
    #                                             downcast='integer')

    # add month
    # transactions['month'] = (transactions.t_dat.dt.month).astype('int8')

    # add year
    # transactions['year'] = (transactions.t_dat.dt.year - 2000).astype('int8')

    # add day of month
    # transactions['day'] = (transactions.t_dat.dt.day).astype('int8')

    # remove t_dat
    transactions.drop(columns=['t_dat'], inplace=True)

    # remove sales_channel_id
    transactions.drop(columns=['sales_channel_id'], inplace=True)

    # print(transactions.head(20))
    return transactions


def pp_data(data, params, force=False, write=True, verbose=True):
    """
    Preprocesses all data
    :param data:
    :param params:
    :param force:
    :param write:
    :param verbose:
    :return:
    """
    articles = data['articles']
    customers = data['customers']
    transactions = data['transactions']
    if verbose:
        print("Preprocessing data... ", end='')
    # redo preprocessing if pickle files are missing
    if not os.path.isfile('pickles/articles.feather') or force:
        articles = pp_articles(articles, transactions, verbose=verbose, params=params)
        if write:
            articles.to_feather('pickles/articles.feather')
    else:
        articles = pd.read_feather('pickles/articles.feather')

    if not os.path.isfile('pickles/customers.pkl') or force:
        customers, cus_keys = pp_customers(customers, transactions)
        if write:
            customers.to_feather('pickles/customers.feather')
            cus_keys.to_feather('pickles/cus_keys.feather')
    else:
        customers = pd.read_feather('pickles/customers.feather')
        cus_keys = pd.read_feather('pickles/cus_keys.feather')

    if not os.path.isfile('pickles/transactions.feather') or force:
        transactions = pp_transactions(transactions)
        if write:
            transactions.to_feather('pickles/transactions.feather')
    else:
        transactions = pd.read_feather('pickles/transactions.feather')

    if verbose:
        print("\r", end='')

    data['articles'] = articles
    data['customers'] = customers
    data['transactions'] = transactions
    data['customer_keys'] = cus_keys
    return data
