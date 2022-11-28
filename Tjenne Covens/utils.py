import numpy as np
import pandas as pd

def map_at_k(actual, predicted, k=12):
    """
    Calculates the mean average precision at k usingthe average precision function below
    :param actual: actual sales
    :param predicted: predicted sales
    :param k: number at which to evaluate the precision
    :return:
    """
    return np.mean([apk(a, p, k) for a, p in list(zip(actual, predicted))])


def apk(actual, predicted, k=12):
    """
    Calculates the average precision at k
    :param actual: actual sales
    :param predicted: predicted sales
    :param k: number at which to evaluate the precision
    :return:
    """

    if len(predicted) == 0:
        return 0
    ap = 0
    correct_at_i = 0
    for i in range(min(k, len(actual))):
        rel = int(actual[i] in predicted)
        correct_at_i += rel
        ap += (correct_at_i / (i+1)) * rel

    return ap/min(k, len(actual))


def map_at_12(samples, predictions):
    """
    Evaluates accuracy of predictor, based on the sales that appeared in the last week of the dataset
    :param samples: Actual sales
    :param predictions: predicted sales
    :return:
    """

    last_week_sales = extract_last_week_sales(samples)
    predictions = filter_predictions(predictions, last_week_sales)
    actual = []
    predicted = []
    customers = list(last_week_sales["customer_id"].unique())
    for customer in customers:
        actual_val = list(last_week_sales.loc[last_week_sales["customer_id"] == customer]["article_id"].head(12))
        predicted_val = list(predictions.loc[predictions["customer_id"] == customer]["article_id"].unique()[0:11])
        actual.append(actual_val)
        predicted.append(predicted_val)

    precision = map_at_k(actual, predicted)
    print(f"map@12 precision: {precision}")     # 0.007
    return precision


def filter_predictions(predictions, last_week_sales):
    predictions_filtered = {}
    for customer in range(len(predictions["customers"])):
        for article in range(len(predictions["prediction"][customer])):
            predictions_filtered[customer*article] = {"customer_id": predictions["customers"][customer],
                                                      "article_id": predictions["prediction"][customer][article]}

    df = pd.DataFrame.from_dict(predictions_filtered, "index")
    df = df.merge(last_week_sales.drop(columns=["customer_id"]), how="inner", on="article_id")
    df.sort_values(by="t_dat", inplace=True)
    return df





def extract_last_week_sales(samples, ordered=True):
    dated_transactions = samples
    if ordered:
        dated_transactions = samples.loc[samples["ordered"] == 1]
    dated_transactions["t_dat"] = pd.to_datetime(dated_transactions["t_dat"])
    latest_date = dated_transactions["t_dat"].max()
    dated_transactions["transaction_age_weeks"] = 52 * (latest_date.year - dated_transactions["t_dat"].dt.year) + 12* (
                latest_date.month - dated_transactions["t_dat"].dt.month) + \
                                                  (latest_date.week - dated_transactions["t_dat"].dt.week)

    last_week_sales = dated_transactions.loc[dated_transactions["transaction_age_weeks"] < 1].sort_values(by="t_dat")
    return last_week_sales.drop(columns=["transaction_age_weeks"])


def extract_all_but_last_week_sales(samples, ordered=True):
    dated_transactions = samples
    if ordered:
        dated_transactions = samples.loc[samples["ordered"] == 1]
    dated_transactions["t_dat"] = pd.to_datetime(dated_transactions["t_dat"])
    latest_date = dated_transactions["t_dat"].max()
    dated_transactions["transaction_age_weeks"] = 52 * (latest_date.year - dated_transactions["t_dat"].dt.year) + 12* (
                latest_date.month - dated_transactions["t_dat"].dt.month) + \
                                                  (latest_date.week - dated_transactions["t_dat"].dt.week)

    last_week_sales = dated_transactions.loc[dated_transactions["transaction_age_weeks"] >= 1]
    return last_week_sales.drop(columns=["transaction_age_weeks"])


def scrape_materials():
    import requests
    import requests
    from bs4 import BeautifulSoup
    import random

    response = requests.get(
        url="https://en.wikipedia.org/wiki/List_of_fabrics",
    )
    soup = BeautifulSoup(response.content, 'html.parser')
    # Get all the links
    allLinks = soup.find(id="bodyContent").find_all("a")
    random.shuffle(allLinks)
    linkToScrape = 0
    materials = []

    for link in allLinks:
        # We are only interested in other wiki articles
        if link['href'].find("/wiki/") == -1:
            continue

        # Use this link to scrape
        materials.append(str(link.contents[0]))

    materials.insert(0, "nylon")
    return materials


def extract_article_material(articles, materials):
    print("extracting materials...")

    articles["material"] = ""
    for index, article in articles.iterrows():
        for material in materials:
            if article["detail_desc"] == 0:
                break
            desc = article["detail_desc"].lower()
            if material in article["detail_desc"]:
                article["material"] = material
                break


        articles[articles["article_id"] == article["article_id"]] = article

    print("extraction done!")

    articles.drop(columns=["detail_desc"], inplace=True)
    return articles

if __name__ == "__main__":
    # a = [[1, 0, 7, 3, 2, 10, 10], [0, 3, 5, 0, 11, 15, 60]]
    # b = [[1, 0, 7, 3, 2, 11, 10], [0, 3, 5, 0, 12, 15, 60]]
    # res = map_at_k(a, b, 12)
    pass
