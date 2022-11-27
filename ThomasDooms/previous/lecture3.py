from random import choices
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_rows = None

articles = pd.read_csv("../data/articles/articles.csv")
# customers = pd.read_csv("data/customers.csv")
# transactions = pd.read_csv("data/transactions.csv")

# customers["single"] = customers.apply(lambda row: row["FN"] != 1.0 and row["Active"] == 1.0, axis=1)
# print(len(customers[customers["single"]]))

# print(articles.groupby(["index_code", "index_name"]).size())

for col in articles.columns:
    if col in ["article_id", "product_code", "prod_name", "detail_desc"]: continue
    print("analyzing column", col, "on articles")
    print(articles[col].describe())
    print(set(articles[col].fillna("")))

# for col in ["club_member_status", "FN", "Active"]:
#     print("analyzing column", col, "on customers")
#     print(customers[col].describe())
#     print(set(customers[col].fillna("")))

# grouped = articles.groupby(by="product_code").size().reset_index(name="count")
# merged = articles.merge(grouped, on="product_code", how="left")
# print(merged[merged["count"] > 1])

# sns.displot(transactions["price"])
# plt.show()

# sns.set_palette(palette)


# plot the distribution of age and amount of transactions
# merged = transactions.merge(customers, on="customer_id", how="left")
# summed = merged.groupby(by="age")["price"].sum().reset_index(name="sum")
#
# print(summed.head(100))
# sns.barplot(data=summed, x='age', y='sum')
# sns.countplot(data=merged, x="age")
# plt.show()


# print(customers["age"].describe())
# print(customers["postal_code"].apply(lambda x: len(x)).describe())


# articles_ids = articles["article_id"]
#
# customers["prediction"] = customers.apply(lambda _: ' '.join([str(x) for x in choices(articles_ids, k=12)]))
# customers[["customer_id", "prediction"]].to_csv("submission.csv", index=False)

# plot the price per article by merging the two datasets
# merged = transactions.merge(articles, on="article_id", how="left")
# summed = merged.groupby(by="article_id")["price"].mean().reset_index(name="mean")
#
# print(summed["mean"].describe())

# sns.displot(summed, x="mean")
# plt.show()

# print('\n\n\n\n')

# plot the mean of purchase price per customer by merging the two datasets
# merged = transactions.merge(customers, on="customer_id", how="left")
# summed = merged.groupby(by="customer_id")["price"].mean().reset_index(name="mean")
#
# print(summed["mean"].describe())

# sns.displot(summed, x="mean")
# plt.show()
