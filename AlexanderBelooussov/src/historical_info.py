import numpy as np
import pandas as pd
import pdcast as pdc

from utils import merge_downcast


def previous_week_customer_info(data_dict):
    """
    Generates customer information: average purchase price, weeks since last purchase
    :param data_dict: dict containing all the relevant data
    :return: Dataframe with columns containing the mentioned info
    """
    print(f"Generating customer statistics", end="")
    transactions_train = data_dict['transactions']

    # =====avg purchase price=====
    avg_purchase_price = transactions_train[['customer_id', 'week', 'price']]
    avg_purchase_price = avg_purchase_price.groupby(['customer_id', 'week'])['price'].sum().rename('price_sum').astype(
        'float32').reset_index()
    avg_purchase_price['purchase_count'] = \
        transactions_train[['customer_id', 'week', 'price']].groupby(['customer_id', 'week'])[
            'price'].count().to_numpy()
    avg_purchase_price['price_cumsum'] = avg_purchase_price.groupby('customer_id')['price_sum'].cumsum()
    avg_purchase_price['purchase_count_cumsum'] = avg_purchase_price.groupby('customer_id')['purchase_count'].cumsum()
    avg_purchase_price['avg_purchase_price'] = (
            avg_purchase_price['price_cumsum'] / avg_purchase_price['purchase_count_cumsum']).astype('float32')
    avg_purchase_price.drop(['price_sum', 'purchase_count', 'price_cumsum', 'purchase_count_cumsum'], axis=1,
                            inplace=True)
    avg_purchase_price['week'] += 1
    customer_week_info = avg_purchase_price.set_index(['customer_id', 'week']).unstack()
    customer_week_info.fillna(method='ffill', inplace=True, axis=1)
    customer_week_info = pdc.downcast(customer_week_info.stack().reset_index())

    # =====weeks since last purchase=====
    purchase_weeks = transactions_train[['customer_id', 'week']].drop_duplicates()
    purchase_weeks['has_purchased'] = int(1)
    purchase_weeks['week'] += 1
    customer_week_info = merge_downcast(customer_week_info, purchase_weeks, on=['customer_id', 'week'], how='left')
    customer_week_info['has_purchased'].fillna(int(0), inplace=True)
    # https://stackoverflow.com/a/44421059
    mask = customer_week_info.groupby('customer_id').has_purchased.cumsum().astype(bool)  # Mask starting zeros as NaN
    customer_week_info = customer_week_info.assign(since_last_purchase=customer_week_info.groupby(
        customer_week_info.has_purchased.astype(bool).cumsum()).cumcount().where(mask))
    customer_week_info['since_last_purchase'] += 1
    customer_week_info['since_last_purchase'].fillna(-1, inplace=True)
    customer_week_info['since_last_purchase'] = customer_week_info['since_last_purchase'].astype('int8')
    customer_week_info.drop('has_purchased', axis=1, inplace=True)
    print(f"\r", end="")
    return customer_week_info


def previous_week_article_info(data_dict):
    """
    Data about sales in previous week(s) for each article and week:
    Sales rank for last week, 4 weeks, and all time.
    Sales for last week, 4 weeks, and all time.
    Average sale price.
    Average buyer age.
    :param data_dict: dict containing all the relevant data
    :return: Dataframe with columns containing the mentioned info
    """
    print(f"Generating article statistics", end="")
    transactions_train = data_dict['transactions']

    # cross join of article ids and weeks
    weeks = transactions_train.week.unique()
    article_ids = transactions_train.article_id.unique()
    weeks, article_ids = np.meshgrid(weeks, article_ids)
    weeks = weeks.flatten()
    article_ids = article_ids.flatten()
    article_week_info = pd.DataFrame({'week': weeks, 'article_id': article_ids})
    customers = data_dict['customers']

    # =====bestseller rank=====
    mean_price = transactions_train \
        .groupby(['week', 'article_id'])['price'].mean()

    weekly_sales_rank = transactions_train \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='min', ascending=False) \
        .rename('bestseller_rank').astype('int16')

    previous_week_rank = merge_downcast(weekly_sales_rank, mean_price, on=['week', 'article_id']).reset_index()
    # previous_week_rank.week += 1
    article_week_info = merge_downcast(article_week_info, previous_week_rank, on=['week', 'article_id'], how='left')
    article_week_info.bestseller_rank.fillna(article_week_info.bestseller_rank.max() + 100, inplace=True)
    article_week_info['bestseller_rank'] = pd.to_numeric(article_week_info['bestseller_rank'], downcast='integer')
    article_week_info["price"] = article_week_info.groupby("article_id")["price"].transform(
        lambda x: x.fillna(x.mean()))

    # =====sales in last week=====
    weekly_sales = transactions_train \
        .groupby('week')['article_id'].value_counts() \
        .rename('1w_sales').astype('int16').reset_index()
    # weekly_sales.week += 1

    article_week_info = pd.merge(article_week_info, weekly_sales, on=['week', 'article_id'], how='left')
    article_week_info['1w_sales'].fillna(0, inplace=True)
    article_week_info['1w_sales'] = pd.to_numeric(article_week_info['1w_sales'], downcast='integer')
    article_week_info = pdc.downcast(article_week_info)

    # =====mean buyer age=====

    mean_buyer_age = merge_downcast(transactions_train[['customer_id', 'article_id', 'week']],
                                    customers[['customer_id', 'age']],
                                    on='customer_id', how='left')
    mean_buyer_age = mean_buyer_age.groupby(['article_id', 'week'])['age'].sum().rename('age_sum').astype(
        'int16').reset_index()
    mean_buyer_age['purchase_count'] = \
        transactions_train[['customer_id', 'article_id', 'week']].groupby(['article_id', 'week'])[
            'customer_id'].count().to_numpy()

    article_week_info = merge_downcast(article_week_info, mean_buyer_age, on=['article_id', 'week'], how='left')
    article_week_info['age_sum'].fillna(0, inplace=True)
    article_week_info['age_sum'] = pd.to_numeric(article_week_info['age_sum'], downcast='integer')
    article_week_info['purchase_count'].fillna(0, inplace=True)
    article_week_info['purchase_count'] = pd.to_numeric(article_week_info['purchase_count'], downcast='integer')
    article_week_info['age_cumsum'] = article_week_info.groupby('article_id')['age_sum'].cumsum()
    article_week_info['purchase_count_cumsum'] = article_week_info.groupby('article_id')['purchase_count'].cumsum()
    article_week_info['purchase_count_cumsum'] += 1
    article_week_info['mean_buyer_age'] = article_week_info['age_cumsum'] / article_week_info['purchase_count_cumsum']
    article_week_info['mean_buyer_age'] = article_week_info['mean_buyer_age'].astype('int16')
    article_week_info.drop(['age_sum', 'purchase_count', 'age_cumsum', 'purchase_count_cumsum'], axis=1, inplace=True)

    # =====sales in last 4 weeks=====
    article_week_info['4w_sales'] = article_week_info.groupby('article_id')['1w_sales'] \
        .rolling(4, min_periods=1).sum().reset_index(0, drop=True)
    article_week_info['4w_sales'] = pd.to_numeric(article_week_info['4w_sales'], downcast='integer')
    # =====sales in all last weeks=====
    article_week_info['all_sales'] = article_week_info.groupby('article_id')['1w_sales'].cumsum()
    article_week_info['all_sales'] = pd.to_numeric(article_week_info['all_sales'], downcast='integer')
    # =====bestseller rank in last 4 weeks=====
    bestseller_4w = article_week_info.groupby('week')['4w_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_4w').astype('int16').reset_index()
    article_week_info['bestseller_4w'] = bestseller_4w['bestseller_4w']
    # =====bestseller rank in all last weeks=====
    bestseller_all = article_week_info.groupby('week')['all_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_all').astype('int16').reset_index()
    article_week_info['bestseller_all'] = bestseller_all['bestseller_all']

    article_week_info['week'] += 1  # shift week so that it is about the previous week
    # print(article_week_info.head(200))
    print(f"\r", end="")
    return article_week_info
