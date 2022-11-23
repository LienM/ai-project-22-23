import pandas as pd

from paths import path

transactions = pd.read_feather(path("transactions", "features"))
# articles = pd.read_feather(path("articles", "features"))
# customers = pd.read_feather(path("customers", "features"))

print(transactions.info())
# print(articles.info())
# print(customers.info())
