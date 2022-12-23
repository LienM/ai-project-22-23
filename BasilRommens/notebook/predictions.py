import pandas as pd
from tqdm import tqdm

from BasilRommens.notebook.notebook import get_cid_2_preds, \
    get_bestseller_dict_article, get_cid_2_age_bin, get_bestseller_dict


def make_and_write_predictions_age(model, test, X_test,
                                   age_bestsellers_previous_week, customers,
                                   customer_encoder, transactions, last_week,
                                   sub_name, is_product_type_recommend=False):
    """
    make predictions for the test set and write them to a csv file, for the age
    bin version
    :param model: the model to make the predictions with
    :param test: the complete test set dataframe
    :param X_test: the X part of the test set dataframe
    :param age_bestsellers_previous_week: the age bestsellers of previous week
    :param customers: customers dataframe
    :param customer_encoder: customer nr to customer id encoder
    :param transactions: transactions dataframe
    :param last_week: last week to make the predictions for
    :param sub_name: the submission file name
    :return: nothing
    """
    # get the predictions and convert to dict
    test['preds'] = model.predict(X_test)
    c_id_2_predicted_article_ids = get_cid_2_preds(test)

    # get the bestsellers for each age bin and convert to a dict
    if is_product_type_recommend:
        bestsellers_age_bin_dict = get_bestseller_dict(
            age_bestsellers_previous_week, transactions, last_week)
    else:
        bestsellers_age_bin_dict = get_bestseller_dict_article(
            age_bestsellers_previous_week, last_week)

    # construct the predictions
    customer_ids = customers['customer_id'].unique()
    sub = pd.DataFrame(
        {'customer_id': customer_encoder.inverse_transform(customer_ids),
         'prediction': ['' for _ in range(len(customer_ids))]})

    # add predictions for the customers and add most popular to them
    preds = []
    cid_2_age_bin = get_cid_2_age_bin(transactions)
    for customer_id in tqdm(customers['customer_id'].unique()):
        # if not in the age bin dict then use the garbage bin
        if customer_id not in cid_2_age_bin.keys():
            customer_age_bin = -1
        else:
            customer_age_bin = cid_2_age_bin.get(customer_id, -1)

        # return the customer specific predictions
        pred = c_id_2_predicted_article_ids.get(customer_id, [])
        # get the bestsellers of the age bin
        bestsellers_age_bin = bestsellers_age_bin_dict.get(customer_age_bin, [])

        # combine custom and bestseller predictions
        pred = pred + bestsellers_age_bin

        # only take the last predictions
        preds.append(pred[:12])

    # convert the predictions to a string
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub['prediction'] = preds

    # name and write the predictions
    sub.to_csv(f'out/{sub_name}.csv.gz', index=False)


def make_and_write_predictions(model, test, X_test, bestsellers_last_week,
                               customers, customer_encoder,
                               sub_name):
    """
    make predictions for the test set and write them to a csv file
    :param model: model to make the predictions with
    :param test: test set
    :param X_test: the X part of the test set
    :param bestsellers_last_week: all the bestsellers of last week
    :param customers: customers dataframe
    :param customer_encoder: customer nr to customer id encoder
    :param sub_name: submission file name
    :return: nothing
    """
    # get the predictions and convert to dict
    test['preds'] = model.predict(X_test)
    c_id2predicted_article_ids = get_cid_2_preds(test)

    # construct the predictions
    customer_ids = customers['customer_id'].unique()
    sub = pd.DataFrame(
        {'customer_id': customer_encoder.inverse_transform(customer_ids),
         'prediction': ['' for _ in range(len(customer_ids))]})

    # add predictions for the customers and add most popular to them
    preds = []
    for customer_id in tqdm(customers['customer_id'].unique()):
        # return the customer specific predictions
        pred = c_id2predicted_article_ids.get(customer_id, [])

        # combine custom and bestseller predictions
        pred = pred + bestsellers_last_week

        # only take the last predictions
        preds.append(pred[:12])

    # convert the predictions to a string
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub['prediction'] = preds

    # name and write the predictions
    sub.to_csv(f'out/{sub_name}.csv.gz', index=False)
