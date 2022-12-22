import pandas as pd
import random
import json

settings_file = "./settings.json"
settings = json.load(open(settings_file))
random.seed(42)

data_dir = settings["data_directory"]
candidate_dir = settings["candidate_directory"]
processed_filenames = settings["data_filenames"]["processed"]
customers_filename = data_dir + processed_filenames["customers"]
articles_filename = data_dir + processed_filenames["articles"]
bestsellers_filename = data_dir + processed_filenames["bestsellers"]
transactions = pd.read_csv(data_dir + processed_filenames["transactions"])
bestsellers = pd.read_csv(bestsellers_filename)
prediction_week = transactions.t_dat.max() + 1
bestsellers = bestsellers[bestsellers["t_dat"] == prediction_week]

"""
Using a sliding window with an age range of 5 (-2 to +2), 
determines the most popular items for each window

One note is that 0 is used as a NaN replacement, this does mean that NaN-ages
are not part of any other window and vice versa, but they do get a separate category this way
"""
print('generating age-dependent candidates')
most_popular_count = settings["popular_candidates"]
age_distance = 2
age_list = pd.read_csv(customers_filename)["age"].sort_values().unique()
articles = pd.read_csv(articles_filename)
article_columns = ["age"] + articles.columns
popular_articles = pd.DataFrame(columns=article_columns)


for age in age_list:
    temp = transactions[transactions["age"] == age].drop_duplicates()
    temp["age_score"] = temp["article_id"].value_counts()
    temp.sort_values("age_score", inplace=True, ascending=False)
    temp = temp.head(most_popular_count)
    temp.reindex(columns=article_columns)
    popular_articles = pd.concat([popular_articles, temp])
    del temp

popular_articles = popular_articles[["age", "article_id"]]
popular_articles = popular_articles.merge(articles, how="left", on="article_id")

counter = 0
customers = pd.read_csv(customers_filename)
"""
For customers of each age, determine the relevant window to look in, and
concatenate all items to the recommendations
"""
for age in age_list:
    print("working for age", age, end="\r")
    customer_temp = customers[customers["age"] == age].copy()
    candidate_filenames = settings["candidate_filenames"]
    age_range = [age + i for i in range(-age_distance, age_distance+1)]
    relevant_articles = popular_articles[popular_articles["age"].isin(age_range)]
    relevant_articles = relevant_articles.drop("age", axis=1)
    customer_temp = customer_temp.merge(relevant_articles, how="cross")
    customer_temp.drop_duplicates(subset=["customer_id", "article_id"], inplace=True)
    customer_temp["sales_channel_id"] = 2
    customer_temp["price_discrepancy"] = customer_temp["average_article_price"] - \
                                         customer_temp["average_customer_budget"]
    customer_temp["t_dat"] = prediction_week
    customer_temp["price"] = customer_temp["average_article_price"]
    customer_temp["ordered"] = 1
    customer_temp = customer_temp.merge(bestsellers, how="left", on=["article_id", "t_dat"])
    customer_temp = customer_temp.reindex(columns=transactions.columns)
    customer_temp["bestseller_rank"].fillna(999, inplace=True)
    customer_temp.to_csv(candidate_dir + str(int(age)) + candidate_filenames["age_based"], index=False)

