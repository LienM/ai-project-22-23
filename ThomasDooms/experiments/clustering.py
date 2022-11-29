import os.path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

from ThomasDooms.src.paths import path
from sklearn.cluster import KMeans

if not os.path.exists(path('articles', 'clustered')):
    articles = pd.read_feather(path('articles', 'features'))

    features = articles[[f"detail_desc_{i}" for i in range(8)]].to_numpy()

    kmeans = KMeans(n_clusters=200).fit(features)
    articles["cluster"] = kmeans.labels_

    articles.to_feather(path('articles', 'clustered'))
else:
    articles = pd.read_feather(path('articles', 'clustered'))

if not os.path.exists("../data/mean.feather"):
    customers = pd.read_feather(path('customers', 'full'))
    transactions = pd.read_feather(path('transactions', 'full'))
    transactions['season'] = (transactions['week'] % 52) // 13

    merged = pd.merge(transactions, articles[["article_id", "cluster"]], on='article_id', how='left')
    merged = pd.merge(merged, customers[["customer_id", "postal_code"]], on='customer_id', how='left')

    # df = merged.groupby(['season', 'cluster']).size().unstack(fill_value=0)
    #
    # sns.heatmap(cosine_similarity(df.to_numpy()))
    # plt.show()

    df = merged.groupby(['season', 'cluster']).size().unstack(fill_value=0)
    df = df.to_numpy()

    factor = np.apply_along_axis(lambda x: (x[2] + x[3]) / sum(x), 0, df)
    print(factor)

    # sns.histplot(factor, bins=10)
    # plt.show()

    labels = pd.DataFrame({'cluster': list(range(200)), 'factor': factor})
    merged = pd.merge(merged, labels, on='cluster', how='left')

    # change the value of factor to (1-x) if spring/summer, x if autumn/winter
    inverse = -2 * (merged['season'] // 2) + 1
    plussed = merged['season'] // 2
    merged['factor'] = plussed + inverse * merged['factor']

    mean = merged.groupby('postal_code')['factor'].mean().sort_values(ascending=False).reset_index(name='mean')
    mean.to_feather("../data/mean.feather")
else:
    mean = pd.read_feather("../data/mean.feather")

print(mean['mean'].describe())
sns.histplot(mean['mean'], bins=50)
plt.show()



