import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ThomasDooms.src.paths import path

# transactions = pd.read_feather(path('transactions', 'full'))
# transactions['season'] = transactions['week'] % 13
# transactions['season'] = transactions['season'].astype('int8')
#
# purchases = transactions.groupby(['article_id', 'season']).size().reset_index(name='purchases')
#
# seasons = pd.merge(transactions, purchases, on=['article_id', 'season'], how='left')
#
# top_sellers = seasons.sort_values(['season', 'purchases'], ascending=False).groupby('season').head(20)
# print(top_sellers)

start = time.time()
transactions = pd.read_feather(path('transactions', 'full'))
transactions['season'] = transactions['week'] // 13

transactions['article_id'] = pd.factorize(transactions['article_id'])[0].astype('int32')

df = transactions.groupby(['season', 'article_id']).size().unstack(fill_value=0)
features = PCA(n_components=8).fit_transform(df.to_numpy())

sns.heatmap(cosine_similarity(features))
plt.show()


differences = [1 - cosine_similarity([first], [second])[0, 0] for first, second in zip(features, features[1:])]

top_k = sorted(enumerate(differences), key=lambda x: x[1], reverse=True)[:5]
print(top_k)

sns.lineplot(x=range(len(features) - 1), y=differences)
plt.show()

# (38, 0.25), (37, 0.23), (62, 0.20), (79, 0.10), (98, 0.09)
# 13 june, 6 june, 28 nov, 26 march, 6 aug

coordinates = TSNE(n_components=2, perplexity=8).fit_transform(features)
coordinates = pd.DataFrame(coordinates, columns=['x', 'y'])
coordinates['week'] = range(len(coordinates))

sns.scatterplot(coordinates['x'], coordinates['y'], hue=coordinates['week'])
plt.show()

# print(x)

# matrix = np.zeros((article_ids, weeks), dtype=np.int32)
#
# for article_id, week in tqdm(zip(transactions['article_id'], transactions['week'])):
#     matrix[article_id, week] += 1

# print(time.time() - start)
# print(matrix)
