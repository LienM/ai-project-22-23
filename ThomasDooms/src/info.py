# ============================================================================
# @author      : Thomas Dooms
# @date        : 14/11/22
# @copyright   : MA2 Computer Science - Thomas Dooms - University of Antwerp
# ============================================================================

import pandas as pd

BASE = "data"

# transactions = pd.read_feather(f"{BASE}/transactions.feather")
articles = pd.read_feather(f"{BASE}/articles.feather")
# customers = pd.read_feather(f"{BASE}/customers.feather")

print(len(articles))
