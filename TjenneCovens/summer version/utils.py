import numpy as np
import pandas as pd
import datetime as dt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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


def transactions_age_in_weeks(samples):
    """
    Extract the age in weeks the age of all transactions. This value can then be used for recent popularity.
    """
    last_sale = samples['t_dat'].max()
    samples['age_in_weeks'] = samples['t_dat'] - last_sale
    samples['age_in_weeks'] = abs(samples['age_in_weeks'].dt.days) // 7
    return samples


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
    with open('./../materials.txt', 'w') as fp:
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
    with open('./../materials.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]

            materials.append(x)

    return materials


wool = ['alpaca', 'mohair']
elastane = ['bikini', 'scuba', 'swimsuit']
fine_knit = ['fine knit', 'fine knit', 'fine knitted']
rib_nit = ['ribknit', 'rib nkit', 'ribknitted']
def check_material(desc, materials):
    """
    Checks which material is mentioned in the description
    """



    if desc == 'none':
        return 'other'

    words = desc.split(' ')
    for word in range(len(words)):
        # check single word
        for material in materials:
            if material in words[word]:
                if material == 'jeans':
                    return 'denim'
                return material

        # check 2-gram
        if word < len(words) - 1:
            combined_words = words[word] + ' ' + words[word+1]
            for material in materials:
                if material == combined_words:
                    return material


    # check secondary references
    for word in range(len(words)):
        if words[word] in wool:
            return 'wool'
        elif words[word] in elastane:
            return 'elastane'
        elif words[word] in fine_knit:
            return 'fine knit'
        elif words[word] in rib_nit:
            return 'rib nit'
        elif 'fastdrying functional fabric' in desc:
            return "fastdrying functional fabric"


    return 'other'

def extract_article_material(articles):
    """
    Extracts the material feature from the detailed description column
    : articles: the articles dataframe
    """
    materials = read_materials()

    articles["material"] = ""
    articles["material"] = articles["detail_desc"].apply(lambda x: check_material(x, materials))
    return articles


def check_season(row):
    """
    Checks which season matches the article best
    : row: article row, containing all counts for the seasons
    """
    from operator import itemgetter

    # set sigma and rho (see report)
    sales_count_boundary = 50          # rho
    season_percentage_boundary = 0.3    # sigma

    # verify if enough sales are made

    if row['total'] > sales_count_boundary:

        # find the highest number
        seasons = ["winter", "summer", "fall", "spring"]
        counts = list(zip(seasons, [row["winter"], row["summer"], row["fall"],
                                    row["spring"]]))

        season = max(counts, key=itemgetter(1))

        # check if this number exceeds lambda
        if season[1] > season_percentage_boundary:
            return season[0]

    return 'all_round'


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
    winter_transactions = winter_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="winter")
    winter_transactions["season"] = "winter"
    summer_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(summer)].drop(columns=["t_dat"])
    summer_transactions = summer_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="summer")
    summer_transactions["season"] = "summer"
    spring_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(spring)].drop(columns=["t_dat"])
    spring_transactions = spring_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="spring")
    spring_transactions["season"] = "spring"
    fall_transactions = dated_transactions.loc[dated_transactions["t_dat"].dt.month.isin(fall)].drop(columns=["t_dat"])
    fall_transactions = fall_transactions["article_id"].value_counts().rename_axis("article_id").reset_index(name="fall")
    fall_transactions["season"] = "fall"


    # assign season to articles
    articles = articles.merge(spring_transactions, on="article_id", how="left")
    articles = articles.merge(fall_transactions, on="article_id", how="left")
    articles = articles.merge(summer_transactions, on="article_id", how="left")
    articles = articles.merge(winter_transactions, on="article_id", how="left")

    # calculate total sales per item
    articles['total'] = articles['summer'] + articles['spring'] + articles['fall'] + articles['winter']

    # calculate percentages
    articles['summer'] = articles['summer'] / articles['total']
    articles['fall'] = articles['fall'] / articles['total']
    articles['winter'] = articles['winter'] / articles['total']
    articles['spring'] = articles['spring'] / articles['total']

    # prepare for season assignment
    articles = articles.fillna(0)
    articles["season"] = None

    # glue season to results
    articles["season"] = articles.T.apply(lambda x: check_season(x))

    articles = articles.drop(columns=["summer", "winter", "fall", "spring", "season_x", "season_y"])
    return articles

def check_price_category(price):
    """
    Checks in which category the price fits
    : price: float representing an article price
    """
    cheap =  0.15
    normal = 0.4
    affordable = 0.6
    expensive = 1

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



    articles["price_cat"] = articles["price"].apply(lambda x: check_price_category(x))

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


def bucket_ages(customers, start=16 , stop=60, bucket_size=10):
    """
    Convert the ages to buckets to provide generalization
    """
    customers['age_cat'] = None
    current = start
    while current <= stop:
        label = f'{current}-{current+10}'
        condition = (customers['age'] >= current) & (customers['age'] < current+10)
        customers.loc[condition, 'age_cat'] = label
        current += 10
    condition = customers['age'] >= 66
    customers.loc[condition, 'age_cat'] = '66+'

    return customers

def advanced_neg_sampling(transactions, articles, sample_count=5, nr_most_recent=3):
    """
    Creates n personalized negative samples based on the m most recent purchases.
    """
    # select relevant attributes
    transactions_selection = transactions[['customer_id', 't_dat', 'article_id']]
    article_selection = articles[['article_id', 'product_type_no', 'material', 'colour_group_code']]
    customers = transactions['customer_id']

    neg_samples = pd.DataFrame(columns=['customer_id', 'article_id'])

    # batch processing
    batchsize = 10000
    for i in range(0, len(customers), batchsize):
        if i % 100000:
            print((f'{i}/{len(customers)}'))

        # create batch
        batch_customers = pd.DataFrame(columns=['customer_id'])
        batch_customers['customer_id'] = customers.iloc[i:i+batchsize]
        batch_transactions = transactions_selection.merge(batch_customers, on='customer_id', how='inner')

        # find most recent purchases for each customer
        last_purchases = batch_transactions.sort_values(by=['t_dat']).groupby(by=['customer_id']).head(nr_most_recent)
        del batch_transactions
        last_purchases = last_purchases.merge(article_selection, on='article_id', how='inner')
        merger = last_purchases.drop(columns=['article_id', 't_dat'])
        last_purchases.drop(columns=['t_dat', 'product_type_no', 'material', 'colour_group_code'], inplace=True)

        # find similar items to serve as negative samples
        similar_articles = merger.merge(article_selection, on=[ 'product_type_no', 'material', 'colour_group_code'], how='inner').drop(columns=['product_type_no', 'material', 'colour_group_code'])
        del merger
        batch_neg_samples = pd.merge(similar_articles, last_purchases, indicator=True, how='outer').drop_duplicates()
        del last_purchases, similar_articles
        batch_neg_samples = batch_neg_samples[batch_neg_samples['_merge'] != 'both'].drop(columns=['_merge'])

        # randomly sample items
        batch_neg_samples = batch_neg_samples.groupby(by=['customer_id']).sample(sample_count, replace=True)
        neg_samples = pd.concat([neg_samples, batch_neg_samples])

    # add information to neg samples
    neg_samples = neg_samples.merge(articles, on='article_id', how='inner')
    num_neg_samples = neg_samples.shape[0]

    real_dates = transactions["t_dat"].unique()
    neg_samples['t_dat'] = np.random.choice(real_dates, size=num_neg_samples)

    article_and_price = transactions[["article_id", "price"]].drop_duplicates("article_id").set_index("article_id").squeeze()
    neg_samples['price'] = article_and_price[neg_samples['article_id']].values
    del article_and_price
    neg_samples['ordered'] = 0
    neg_samples['sales_channel_id'] = np.random.choice([1, 2], size=num_neg_samples)

    return neg_samples



if __name__ == "__main__":
    import string
    articles = pd.read_csv('../data/articles_sample5.csv.gz')
    customers = pd.read_csv('../data/customers_sample5.csv.gz')
    transactions = pd.read_csv('../data/transactions_sample5.csv.gz')
    article_id_encoder = preprocessing.LabelEncoder()
    transactions['article_id'] = article_id_encoder.fit_transform(transactions['article_id'])
    articles['article_id'] = article_id_encoder.fit_transform(articles['article_id'])

    # transactions = transactions_age_in_weeks(transactions)
    # weekly_averages = transactions.groupby(by=['age_in_weeks', 'price'])

    articles['detail_desc'] = articles['detail_desc'].fillna('none')

    # lowercasing detailed description
    articles['detail_desc'] = articles['detail_desc'].str.lower()

    # removing punctuation from detailed description
    punct_chars = set(string.punctuation)
    articles['detail_desc'] = articles['detail_desc'].apply(
        lambda x: ''.join(char for char in x if char not in punct_chars))
    articles = extract_article_material(articles)
    # articles = extract_season(transactions, articles)
    a = 4
    # articles =  extract_price_category(transactions, articles)
    advanced_neg_sampling(transactions, articles)
    print(articles['season'].value_counts())
