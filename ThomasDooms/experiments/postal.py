import pandas as pd

from ThomasDooms.src.paths import path

customers = pd.read_feather(path("customers", 'features'))

# print(len(set(customers["postal_code"])))
print(customers["postal_code"].value_counts())
print(customers["postal_code"].value_counts().to_frame()["postal_code"].describe())
