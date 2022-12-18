import pandas as pd

BASE = "data"

# transactions = pd.read_feather(f"{BASE}/transactions.feather")
articles = pd.read_feather(f"{BASE}/articles.feather")
# customers = pd.read_feather(f"{BASE}/customers.feather")

print(len(articles))
