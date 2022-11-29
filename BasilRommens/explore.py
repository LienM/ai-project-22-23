import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

from BasilRommens.dataset import part_data_set, read_data_set

articles, customers, transactions = read_data_set('feather')
# articles, customers, transactions = part_data_set('5')
customer_encoder = joblib.load('../data/customer_encoder.joblib')

# taking the relevant columns of all dataframes
relevant_article_cols = ['article_id', 'product_type_no',
                         'graphical_appearance_no', 'colour_group_code',
                         'perceived_colour_value_id',
                         'perceived_colour_master_id', 'department_no',
                         'index_group_no', 'section_no', 'garment_group_no']
relevant_customer_cols = ['customer_id', 'FN', 'Active',
                          'fashion_news_frequency', 'age', 'postal_code']
relevant_transaction_cols = ['t_dat', 'customer_id', 'article_id', 'price',
                             'sales_channel_id']
articles = articles[relevant_article_cols]
customers = customers[relevant_customer_cols]
transactions = transactions[relevant_transaction_cols]

# data set construction by taking only last week of transactions
transactions['week'] = transactions['t_dat'].dt.isocalendar().year * 53 + \
                       transactions['t_dat'].dt.isocalendar().week
transactions['week'] = rankdata(transactions['week'], 'dense')

transactions = transactions.merge(articles, on='article_id')
del articles
transactions = transactions.merge(customers, on='customer_id')
del customers
transactions = transactions.reset_index(drop=True)

transactions = transactions.groupby('age')['product_type'] \
    .count() \
    .rename('count') \
    .reset_index()

fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=transactions, x='age', y='count', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
plt.show()
