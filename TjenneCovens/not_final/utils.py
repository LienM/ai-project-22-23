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
    print(f"\nmap@12 precision: {precision}")
    return precision


def filter_predictions(predictions, last_week_sales):
    """
    Retrieves dataframe containing the articles mentioned in the predictions
    """
    predictions_filtered = {}
    for customer in range(len(predictions["customers"])):
        for article in range(len(predictions["prediction"][customer])):
            predictions_filtered[customer*article] = {"customer_id": predictions["customers"][customer],
                                                      "article_id": predictions["prediction"][customer][article]}

    df = pd.DataFrame.from_dict(predictions_filtered, "index")
    df = df.merge(last_week_sales.drop(columns=["customer_id"]), how="inner", on="article_id")
    df.sort_values(by="t_dat", inplace=True)
    return df



def split_samples(samples, weeks=2):
    """
    Splits the samples for test and train sets based on the age of the transaction
    : samples: dataframe containing the samples
    : weeks: the number of weeks passed, used to split the set on
    """
    samples["t_dat"] = pd.to_datetime(samples["t_dat"])
    latest_date = samples["t_dat"].max()
    samples["transaction_age_weeks"] = 52 * (latest_date.year - samples["t_dat"].dt.year) + 12 * (
            latest_date.month - samples["t_dat"].dt.month) + (latest_date.week - samples["t_dat"].dt.week)

    lastest_sales = samples.loc[samples["transaction_age_weeks"] < weeks].sort_values(by="t_dat")
    other_weeks = samples.loc[samples["transaction_age_weeks"] >= weeks].sort_values(by="t_dat")
    return lastest_sales, other_weeks


def extract_last_week_sales(samples, ordered=True):
    """
    Extracts all sales that occurred in the last week
    """
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
    """
    Extract all transactions that did not occur in the last week
    """
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
    """
    Scrape materials from the wikipedia list
    DISCLAIMER: this function does not clean the scraped materials, this has to be done manually!
    """
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
        materials.append(str(link.contents[0]).lower())

    materials.insert(0, "nylon")
    materials.insert(0, "microfibre")

    # writes materials to txt file
    with open('data/materials.txt', 'w') as fp:
        for material in materials:
            # write each item on a new line
            fp.write("%s\n" % material)
        print('Done')
    return materials


def read_materials():
    """
    Reads the materials in from the material file
    """
    materials = []
    with open('../materials.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]

            materials.append(x)

    return materials


def check_material(desc, materials):
    """
    Checks which material is mentioned in the description
    """
    for material in materials:
        if desc == 0:
            break
        desc = desc.lower()
        if material in desc:
            return material

    return "jersey"

def extract_article_material(articles):
    """
    Extracts the material feature from the detailed description column
    : articles: the articles dataframe
    """
    materials = read_materials()

    articles["material"] = ""
    articles["material"] = articles["detail_desc"].apply(lambda x: check_material(x, materials))

    articles.drop(columns=["detail_desc"], inplace=True)
    return articles


def check_season(row):
    """
    Checks which season matches the article best
    : row: article row, containing all counts for the seasons
    """
    from operator import itemgetter

    seasons = ["winter", "summer", "fall", "spring"]
    counts = list(zip(seasons, [row["count_winter"], row["count_summer"], row["count_fall"],
                                row["count_spring"]]))
    return max(counts, key=itemgetter(1))[0]


def extract_season(transactions, articles):
    """
    Extracts the season feature based on when an article is sold most
    : articles: the articles dataframe
    : transactions: the transactions dataframe
    """
    # define seasons
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    fall = [9, 10, 11]
    winter = [12, 1, 2]
    # get dated articles
    dated_transactions = transactions[["article_id", "t_dat"]].copy()
    dated_transactions["t_dat"] = pd.to_datetime(dated_transactions["t_dat"])

    # find season of transactions
    winter_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(winter)].drop(columns=["t_dat"])
    winter_transactions = winter_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="count_winter")
    winter_transactions["season"] = "winter"
    summer_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(summer)].drop(columns=["t_dat"])
    summer_transactions = summer_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="count_summer")
    summer_transactions["season"] = "summer"
    spring_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(spring)].drop(columns=["t_dat"])
    spring_transactions = spring_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="count_spring")
    spring_transactions["season"] = "spring"
    fall_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(fall)].drop(columns=["t_dat"])
    fall_transactions = fall_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="count_fall")
    fall_transactions["season"] = "fall"

    # assign season to articles
    articles = articles.merge(spring_transactions, on="article_id", how="left")
    articles = articles.merge(fall_transactions, on="article_id", how="left")
    articles = articles.merge(summer_transactions, on="article_id", how="left")
    articles = articles.merge(winter_transactions, on="article_id", how="left")
    articles = articles.fillna(0)
    articles["season"] = None

    # glue season to results
    articles["season"] = articles.T.apply(lambda x: check_season(x))

    articles = articles.drop(columns=["count_summer", "count_winter", "count_fall", "count_spring", "season_x", "season_y"])
    return articles

def check_price_category(price):
    """
    Checks in which category the price fits
    : price: float representing an article price
    """
    cheap =  2.7
    normal = 4
    affordable = 5
    expensive = 9

    if price < cheap:
        return "cheap"

    if price < normal:
        return "normal"

    if price < affordable:
        return "affordable"

    if price <= expensive:
        return "expensive"


def extract_price_category(transactions, articles):
    """
    Extracts the price category feature
    : articles: the articles dataframe
    : transactions: the transactions dataframe
    """


    prices = transactions[["article_id", "price"]].groupby(["article_id"]).min("price")

    # transforms the price to provide better boundaries and scaling
    prices["price"] = -np.log(prices["price"])
    articles = articles.merge(prices, on=["article_id"], how="inner")
    articles["price_cat"] = articles["price"].apply(lambda x: check_price_category(x))
    articles.drop(columns=["price"], inplace=True)

    # one hot encode it all!
    articles = pd.get_dummies(articles, columns=["price_cat"])
    return articles



def combine_features(df, feature_1, feature_2):
    """
    Basic feature combining function
    : df: the dataframe containing the features
    : feature_1: the column name of the first feature
    : feature_2: the column name of the second feature
    """
    df[f"{feature_1}_{feature_2}"] = (df[feature_1].astype(str) + '_' + df[feature_2].astype(str))
    return df


