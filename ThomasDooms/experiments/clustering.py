import os.path

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

from ThomasDooms.src.paths import path
from sklearn.cluster import KMeans

# Cluster on detail description embeddings if the file does not yet exist,
# I use 200 clusters as I think this is about the number of unique categories that I can think of for clothes.
# This is just a guess though, I have not done an extensive analysis.
if not os.path.exists(path('articles', 'clustered')):
    articles = pd.read_feather(path('articles', 'features'))

    features = articles[[f"detail_desc_{i}" for i in range(8)]].to_numpy()

    kmeans = KMeans(n_clusters=200).fit(features)
    articles["cluster"] = kmeans.labels_

    articles.to_feather(path('articles', 'clustered'))
else:
    articles = pd.read_feather(path('articles', 'clustered'))

# Join the clusters to the transactions
transactions = pd.read_feather(path('transactions', 'full'))
merged = pd.merge(transactions, articles[["article_id", "cluster"]], on='article_id', how='left')

# This is a way to get a count table for the 2 columns
# With count table I mean, the amount occurrences of each combination of the 2 columns
# In this case, each cell contains the amount of articles belonging to a cluster that were bought in a certain season
df = merged.groupby(['week', 'cluster']).size().unstack(fill_value=0)
df = df.to_numpy()

# We plot the heatmap of this count table
sns.heatmap(cosine_similarity(df))
plt.show()

# Reduce the dimensionality of the count table to 2 dimensions with t-SNE, a visually pleasing visualization method
# Then get the cosine similarity between each sequential week
features = TSNE(n_components=2).fit_transform(df)
differences = [1 - cosine_similarity([first], [second])[0, 0] for first, second in zip(features, features[1:])]

# Print the weeks in which the change was highest
top_k = sorted(enumerate(differences), key=lambda x: x[1], reverse=True)[:5]
print(top_k)

# And then plot this as a line graph
sns.lineplot(x=range(len(features) - 1), y=differences)
plt.show()

# Convert the numpy to a pandas dataframe for plotting
coordinates = pd.DataFrame(features, columns=['x', 'y'])
coordinates['week'] = range(len(coordinates))

# Scatter with the week as the color
sns.scatterplot(x=coordinates['x'], y=coordinates['y'], hue=coordinates['week'])
plt.show()

# Analysis the postal code
if not os.path.exists("../data/mean.feather"):
    # Read the data we require
    customers = pd.read_feather(path('customers', 'full'))
    transactions = pd.read_feather(path('transactions', 'full'))
    transactions['season'] = (transactions['week'] % 52) // 13

    # Merge all the stuff we need and some more
    merged = pd.merge(transactions, articles[["article_id", "cluster"]], on='article_id', how='left')
    merged = pd.merge(merged, customers[["customer_id", "postal_code"]], on='customer_id', how='left')

    # df = merged.groupby(['season', 'cluster']).size().unstack(fill_value=0)
    #
    # sns.heatmap(cosine_similarity(df.to_numpy()))
    # plt.show()

    # Get the mean of the clusters per season per postal code, same as above
    df = merged.groupby(['season', 'cluster']).size().unstack(fill_value=0)
    df = df.to_numpy()

    # Calculate our metric (summer + spring) / sum(seasons), which is done for each cluster
    factor = np.apply_along_axis(lambda x: (x[2] + x[3]) / sum(x), 0, df)
    print(factor)

    # sns.histplot(factor, bins=10)
    # plt.show()

    # Merge the metric per cluster into the dataframe
    labels = pd.DataFrame({'cluster': list(range(200)), 'factor': factor})
    merged = pd.merge(merged, labels, on='cluster', how='left')

    # change the value of factor to (1-x) if spring/summer, x if autumn/winter
    inverse = -2 * (merged['season'] // 2) + 1
    plussed = merged['season'] // 2
    merged['factor'] = plussed + inverse * merged['factor']

    # Save intermediate results
    mean = merged.groupby('postal_code')['factor'].mean().sort_values(ascending=False).reset_index(name='mean')
    mean.to_feather("../data/mean.feather")
else:
    mean = pd.read_feather("../data/mean.feather")

# Print and plot the desired data
print(mean['mean'].describe())
sns.histplot(mean['mean'], bins=50)
plt.show()



