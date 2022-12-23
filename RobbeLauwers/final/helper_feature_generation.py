
import copy
import pandas as pd
import itertools
import time

def get_purchase_rank_df_of_attributes(transactions,articles,attributes_columns_names,feature_name):
    """
    Given customer ids and arbitrary article features (except article id), returns a df with rows containing each combination of customer_id and combination of
    For example: if attributes_columns_names contains ["garment_group_name"], then the final dataframe will contain for each customer how often he bought a garment with each possible value in garment_group_name, and a rank of which ones are his favourites.
    :param transactions: pandas dataframe: Transactions on which to calculate these features, can be full transactions dataset even if training data is a subset
    :param articles: pandas dataframe: Articles table, should be full table
    :param attributes_columns_names: List of strings: Article Column names for which to calculate these features. Should not contain "article_id"
    :param feature_name: string: Name for the new feature
    :return: pandas dataframe: columns customer_id, attributes_columns_names, feature_name, str(feature_name)+"_rank"
    """

    # To make merges later on easier, this variable contains the article columns asked form in the function argument plus article_id
    attributes_columns_names_plus_article_id = copy.deepcopy(attributes_columns_names)
    attributes_columns_names_plus_article_id.insert(0,"article_id")

    # To make merges later on easier, this variable contains the article columns asked form in the function argument plus customer_id
    attributes_columns_names_plus_customer_id = copy.deepcopy(attributes_columns_names)
    attributes_columns_names_plus_customer_id.insert(0,"customer_id")

    # From articles, select only relevant columns. If we want to calculate what a users favourite colour is, we do not need the garment type.
    articles_selected = articles[attributes_columns_names_plus_article_id]

    # This merge results in a dataframe containing for each transaction from the function argument the customer_id, article_id and article features as given in the attributes_columns_names argument
    big_df = pd.merge(articles_selected,transactions[["customer_id","article_id"]],on=["article_id"])

    # Adds a column containing for each transaction how often the customer has already bought clothing with the same attributes_columns_names as the article_id from the transaction
    big_df = big_df.groupby(attributes_columns_names_plus_customer_id).size().reset_index(name=feature_name)

    # Adds a column containing for each transaction the rank that the user gives to clothing with the same attributes_columns_names as the article_id in the transaction.
    # In this case, rank means that if the article id is blue and the user bought lots of blue things, it will be one. If the article is red and red is the users second favourite, it will be 2 etc
    big_df[feature_name + "_rank"] =  big_df.groupby("customer_id")[feature_name].rank(method="dense",ascending=False)
    return big_df

def get_purchase_count_df_of_attributes(transactions,articles,attributes_columns_names,feature_name):
    """
    Given customer ids and arbitrary article features (except article id), returns a df with rows containing each combination of customer_id and combination of
    For example: if attributes_columns_names contains ["garment_group_name"], then the final dataframe will contain for each customer how often he bought a garment with each possible value in garment_group_name, and a rank of which ones are his favourites.
    :param transactions: pandas dataframe: Transactions on which to calculate these features, can be full transactions dataset even if training data is a subset
    :param articles: pandas dataframe: Articles table, should be full table
    :param attributes_columns_names: List of strings: Article Column names for which to calculate these features. Should not contain "article_id"
    :param feature_name: string: Name for the new feature
    :return: pandas dataframe: columns customer_id, attributes_columns_names, feature_name, str(feature_name)+"_rank"
    """

    # To make merges later on easier, this variable contains the article columns asked form in the function argument plus article_id
    attributes_columns_names_plus_article_id = copy.deepcopy(attributes_columns_names)
    attributes_columns_names_plus_article_id.insert(0,"article_id")

    # To make merges later on easier, this variable contains the article columns asked form in the function argument plus customer_id
    attributes_columns_names_plus_customer_id = copy.deepcopy(attributes_columns_names)
    attributes_columns_names_plus_customer_id.insert(0,"customer_id")

    # From articles, select only relevant columns. If we want to calculate what a users favourite colour is, we do not need the garment type.
    articles_selected = articles[attributes_columns_names_plus_article_id]

    # This merge results in a dataframe containing for each transaction from the function argument the customer_id, article_id and article features as given in the attributes_columns_names argument
    big_df = pd.merge(articles_selected,transactions[["customer_id","article_id"]],on=["article_id"])

    # Adds a column containing for each transaction how often the customer has already bought clothing with the same attributes_columns_names as the article_id from the transaction
    big_df = big_df.groupby(attributes_columns_names_plus_customer_id).size().reset_index(name=feature_name)

    # Adds a column containing for each transaction the rank that the user gives to clothing with the same attributes_columns_names as the article_id in the transaction.
    # In this case, rank means that if the article id is blue and the user bought lots of blue things, it will be one. If the article is red and red is the users second favourite, it will be 2 etc
    # big_df[feature_name + "_rank"] =  big_df.groupby("customer_id")[feature_name].rank(method="dense",ascending=False)
    return big_df

def add_features_to_data(article_features,do_combinations_of_features,columns_to_use,transactions_full,featuresBackXWeeks,articles,min_week,test_week,use_count=True,use_rank=True,verbose=False):
    # For features generated by get_purchase_rank_df_of_attributes
    # Key: feature name
    # Value: list of strings: columns of article_ids the feature is based on
    new_features = dict()
    # For each column listed in article_features: say that we want to make a new feature out of it later on
    for feature_column in article_features:
        new_features["amount_of_(" + feature_column + ")"] = [feature_column]
    # For each combination of 2 columns listed in article_features: say that we want to make a new feature out of it
    # later on
    if do_combinations_of_features:
        for double_features in itertools.combinations(article_features, 2):
            new_features["amount_of_(" + double_features[0] + "_" + double_features[1] + ")"] = [double_features[0],
                                                                                                 double_features[1]]

    all_new_features = []

    # Lecture 6
    # For everything I said I would make a new feature of:
    for feature_name, partial_columns in new_features.items():
        tempname = str(feature_name) + "_temp"
        time_start = time.time()
        # Tell ranker to use new features
        # Can be commented out to only use either count/rank
        if use_count:
            columns_to_use.append(feature_name)
        if use_rank:
            columns_to_use.append(feature_name + "_rank")

        current_week = min_week + 1
        feature_all_weeks = pd.DataFrame()
        while current_week < test_week:
            # Features are calculated on purchase history: taki into account not to incorporate future data
            if feature_all_weeks.empty:
                # See definition of get_purchase_rank_df_of_attributes for comments
                # Get purchase count of articles with certain attributes in past X weeks
                df_with_customer_id_and_features_and_count = get_purchase_count_df_of_attributes(transactions_full[(
                                                                                                                               transactions_full.week < current_week) & (
                                                                                                                               transactions_full.week > current_week - featuresBackXWeeks)],
                                                                                                 articles,
                                                                                                 partial_columns,
                                                                                                 feature_name)
                df_with_customer_id_and_features_and_count[feature_name] = df_with_customer_id_and_features_and_count[
                    feature_name].fillna(0)  # If user did not buy anything yet, set purchase count to 0
                df_with_customer_id_and_features_and_count["week"] = current_week
            else:
                # Objective of this section: calculate purchase counts for one week, and add those of last week to
                # prevent calculating the entirety of get_purchase_count_df_of_attributes for the (near) full dataset
                # every week
                # Get purchase counts for this week
                df_with_customer_id_and_features_and_count = get_purchase_count_df_of_attributes(
                    transactions_full[(transactions_full.week == current_week)], articles, partial_columns,
                    feature_name)
                df_with_customer_id_and_features_and_count[feature_name] = df_with_customer_id_and_features_and_count[
                    feature_name].fillna(0)  # If user did not buy anything this week, set purchase count to 0
                # get purchase counts of last week
                previous_week_purchase_counts = feature_all_weeks[(feature_all_weeks.week == (current_week - 1))]
                # Rename purchase counts of this week
                df_with_customer_id_and_features_and_count = df_with_customer_id_and_features_and_count.rename(
                    columns={feature_name: tempname})
                # For each customer and selected article feature, get purchase count of this one week and purchase
                # counts up to and including previous week
                df_with_customer_id_and_features_and_count = pd.merge(
                    df_with_customer_id_and_features_and_count[["customer_id", tempname] + partial_columns],
                    previous_week_purchase_counts[["customer_id", feature_name] + partial_columns], how="outer",
                    on=[["customer_id"] + partial_columns][0])
                # If customer had not purchased items with certain feature in either this week or before this week,
                # set purchase count to 0
                df_with_customer_id_and_features_and_count.fillna(0)
                # Add purchase counts of this week and before this week to get purchase counts up to and including
                # this week
                df_with_customer_id_and_features_and_count[feature_name] = df_with_customer_id_and_features_and_count[
                                                                               feature_name] + \
                                                                           df_with_customer_id_and_features_and_count[
                                                                               tempname]
                # Remove temporary column used for calculation above
                df_with_customer_id_and_features_and_count.drop(columns=[tempname], inplace=True)
                # All purchase counts are up to and including this week
                df_with_customer_id_and_features_and_count["week"] = current_week

            # Store all weeks in one dataframe
            if feature_all_weeks.empty:
                feature_all_weeks = df_with_customer_id_and_features_and_count.copy()
            else:
                feature_all_weeks = pd.concat([feature_all_weeks, df_with_customer_id_and_features_and_count])
            current_week += 1

        # Include ranking of feature: if blue was the users most bought garment color, each transaction where the customer buys blue things will be 1
        feature_all_weeks[feature_name + "_rank"] = feature_all_weeks.groupby(["customer_id", "week"])[
            feature_name].rank(method="dense", ascending=False)

        # Keep list of all new feature dataframes + column names to merge them later
        all_new_features.append([feature_all_weeks, partial_columns])

        # Print time it took to generate feature
        if verbose:
            print(feature_name + str(time.time() - time_start))

    return all_new_features