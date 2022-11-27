import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from karateclub import DeepWalk, Diff2Vec, Node2Vec
from matplotlib import pyplot as plt
import time
from itertools import product

from ThomasDooms.src.paths import path

start = time.time()
transactions = pd.read_feather(path('transactions', 'sample'))
articles = pd.read_feather(path('articles', 'full'))
customers = pd.read_feather(path('customers', 'full'))

customers = customers[:10_000]
articles = articles[:10_000]

print(f"done reading data in {time.time() - start} seconds")

start = time.time()
B = nx.Graph()

a_len = len(articles)
c_len = len(customers)

a_map = {a_id: idx for idx, a_id in enumerate(articles['article_id'].values)}
c_map = {c_id: idx + a_len for idx, c_id in enumerate(customers['customer_id'].values)}

# transactions["article_id"] = transactions["article_id"].map(a_map)
# transactions["customer_id"] = transactions["customer_id"].map(c_map)

B.add_nodes_from(range(a_len), bipartite=0)
B.add_nodes_from(range(a_len, a_len+c_len), bipartite=1)

B.add_edges_from([(x, y) for x, y in product(range(50), range(a_len + 50))])
# B.add_edges_from(transactions[["article_id", "customer_id"]].values)
print(f"done constructing graph in {time.time() - start} seconds")

# print(B.nodes(data=True)[:10])

# check if there are duplicates between customer ids and article ids
# print(len(set(customers["customer_id"].values).intersection(set(articles["article_id"].values))))

start = time.time()
model = DeepWalk(walk_number=3, walk_length=10, dimensions=16, workers=8)
model.fit(B)
embedding = model.get_embedding()
print(f"done embedding graph in {time.time() - start} seconds")
# print(embedding)

