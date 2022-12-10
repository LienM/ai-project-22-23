import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""
Prepare the data for LSTM training.
"""

if not os.path.isdir('../../data/lstm'):
    os.mkdir('../../data/lstm')

# read dataframes, only keep customer_id and article_id
transactions: pd.DataFrame = pd.read_csv('../../data/transactions_train.csv', dtype=object)
transactions = transactions[['customer_id', 'article_id']].copy()

# concatenate and count article_ids
encoder = LabelEncoder()
transactions['article_id'] = encoder.fit_transform(transactions['article_id'])
transactions['customer_id'] = encoder.fit_transform(transactions['customer_id'])
transactions['article_id'] = transactions['article_id'].astype(str)
transactions = transactions.groupby(['customer_id'], as_index=False).agg({'article_id': ' '.join})
transactions['num_article_ids'] = transactions['article_id'].str.count('\s') + 1

# find users that did not purchase any article
customers: pd.DataFrame = pd.read_csv('../../data/customers.csv')
customers = customers[['customer_id']].copy()
customers['article_id'] = ''
customers['num_article_ids'] = 0
customers['customer_id'] = encoder.fit_transform(customers['customer_id'])
transactions = pd.concat([transactions, customers], ignore_index=True)
transactions = transactions.drop_duplicates(subset=['customer_id'], keep='first')
transactions_0_articles = transactions[transactions["num_article_ids"] == 0]
print('Nr of users that did not purchase any article:', transactions_0_articles.shape[0], end=' ')
print(f'({round((transactions_0_articles.shape[0] / customers.shape[0]) * 100, 2)}%)')

# split transactions
transactions = transactions.sort_values(by=['num_article_ids'])
transactions_lt_3_articles = transactions[transactions['num_article_ids'].isin([1, 2])]
transactions_gte_3_articles = transactions[transactions['num_article_ids'] >= 3]
print('Nr of users that purchased < 3 articles:', transactions_lt_3_articles.shape[0], end=' ')
print(f'({round(((transactions_lt_3_articles.shape[0]) / customers.shape[0]) * 100, 2)}%)')
print('Nr of users that purchased >= 3 articles:', transactions_gte_3_articles.shape[0], end=' ')
print(f'({round(((transactions_gte_3_articles.shape[0]) / customers.shape[0]) * 100, 2)}%)')

# remove the customer with 1895 purchases
transactions_eq_1895_articles = transactions_gte_3_articles[transactions_gte_3_articles['num_article_ids'] == 1895]
transactions_gte_3_articles = transactions_gte_3_articles[transactions_gte_3_articles['num_article_ids'] < 1895]
print('Nr of users that purchased >= 3 articles:', transactions_gte_3_articles.shape[0], end=' ')
print(f'({round(((transactions_gte_3_articles.shape[0]) / customers.shape[0]) * 100, 2)}%)')

# frames to file
transactions.to_csv('../../data/lstm/transactions.csv', index=False)
transactions_gte_3_articles.to_csv('../../data/lstm/transactions_gte_3_articles.csv', index=False)
transactions_lt_3_articles.to_csv('../../data/lstm/transactions_lt_3_articles.csv', index=False)
transactions_0_articles.to_csv('../../data/lstm/transactions_0_articles.csv', index=False)
transactions_eq_1895_articles.to_csv('../../data/lstm/transactions_eq_1895_articles.csv', index=False)

# define vocabulary size
vocabulary_size = transactions['article_id'].nunique() + 1
print('Vocabulary size:', vocabulary_size)

# pad sequences
max_len = transactions_gte_3_articles['num_article_ids'].max()
print('Max # purchases by single user:', max_len)

json.dump({'vocabulary_size': vocabulary_size, 'max_len': max_len}, open('../../data/lstm/parameters.json', 'w'))
