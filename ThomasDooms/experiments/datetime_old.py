import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_rows = None

transactions = pd.read_csv("data/transactions.csv")
transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], format='%Y-%m-%d')

print(transactions["t_dat"].min())
print(transactions["t_dat"].max())

customers = pd.read_csv("data/customers.csv")
customers["summed"] = customers.groupby("age")["age"].count()
print(customers)
