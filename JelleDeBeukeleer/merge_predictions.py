import pandas as pd


"""
Simple script, merges all intermediate predictions files to one final predictions.csv
"""
df = pd.DataFrame(columns=["customer_id", "prediction"])

for i in range(5):
    print("merging predictions", i)
    name = "../data/" + str(i) + "predictions.csv"
    new_df = pd.read_csv(name)
    df = pd.concat([df, new_df])

df.to_csv("predictions.csv", index=False)