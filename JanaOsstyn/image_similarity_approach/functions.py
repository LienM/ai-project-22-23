import shutil, os, cv2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

INPUT_DATA_PATH = "../../data"
OUTPUT_DATA_PATH = "../../data_output"


def idp(filename):
    """
    Add the input data path to the filename
    :param filename: filename of file in the input data directory
    :return: the path to the file
    """
    return f'{INPUT_DATA_PATH}/{filename}'


def odp(filename):
    """
    Add the output data path to the filename
    :param filename: filename of file in the output data directory
    :return: the path to the file
    """
    return f'{OUTPUT_DATA_PATH}/{filename}'


def move_files(filenames, dir_from, dir_to):
    """
    Move the specified filenames to another directory
    :param filenames: filenames (without path)
    :param dir_from: source directory
    :param dir_to: destination directory
    """
    for filename in filenames:
        shutil.move(f'{dir_from}/{filename}', f'{dir_to}/{filename}')


def create_directory_if_not_exists(directory):
    """
    Create a directory with the given name in the output path
    :param directory: name of the new directory
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)


def remove_directory_if_exists(directory):
    """
    Remove the directory with given name from the output path
    :param directory: name of the directory to remove
    """
    if os.path.isdir(directory):
        shutil.rmtree(directory)


def list_directory_if_exists(directory):
    """
    If the given directory exists, list it's content. If not, return an empty list.
    :param directory: the name of the directory
    :return: a list containing the content of the directory
    """
    if os.path.isdir(directory):
        return sorted(os.listdir(directory))
    else:
        return []


def load_img(filename):
    """
    Load the image from the file with given name
    :param filename: filename of the image
    :return: the image
    """
    img = cv2.imread(idp(f'images/{filename}'))
    if img is None:
        return None
    w, h, _ = img.shape
    resized_img = cv2.resize(img, (int(h * 0.1), int(w * 0.1)), interpolation=cv2.INTER_AREA)
    return resized_img


def plot_img_grid(fig_dict, n_rows = 1, n_cols=1, fig_size=(4, 4)):
    """
    Plot a grid of n_rows x n_cols images
    :param fig_dict: a dictionary mapping image names (titles for each image) to the actual images
    :param n_rows: nr of rows for the grid
    :param n_cols: nr of columns for the grid
    :param fig_size: size of a single image within a grid
    """
    fig, ax_list = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=fig_size)
    for ind,title in enumerate(fig_dict):
        try:
            ax_list.ravel()[ind].imshow(cv2.cvtColor(fig_dict[title], cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        ax_list.ravel()[ind].set_title(title)
    for ind in range(n_rows * n_cols):
        ax_list.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    return


def article_list_to_str(article_id_list):
    """
    Convert the list of article_ids to a space-separated string while preserving order
    :param article_id_list: a list of article_ids
    :return: a string with space-separated article_ids
    """
    return ' '.join(article_id_list)


def article_str_to_list(article_id_str):
    """
    Convert the string of space-separated article_ids to a list while preserving order
    :param article_id_str: a string with space-separated article_ids
    :return: a list of article_ids
    """
    return article_id_str.split(' ')


def get_similar_article_dict(similarity_df_article_ids, similarity_levels=None):
    """
    Map each article_id to a tuple of similar articles, where similarity_levels[i] indicates the 'level of similarity'
    of result[article_id][i]. For example, if similarity_levels == [2, 5, 7], then article_id xxxx maps to
    (yyyy, zzzz, aaaa), where yyyy is the 2nd similar article to xxxx, zzzz the 5th and aaaa the 7th.
    :param similarity_df_article_ids: a dataframe containing the top-100 most similar articles for each article
    :param similarity_levels: a list (or tuple) of similarity levels
    :return: the resulting dictionary
    """
    original_article_ids = similarity_df_article_ids['article_index'].tolist()
    similar_article_ids = list(zip(
        *[similarity_df_article_ids[str(sim_level)].tolist() for sim_level in similarity_levels]
    ))
    result = {original_article_ids[i]: similar_article_ids[i] for i in range(len(original_article_ids))}
    return result


def get_similar_score_dict(similarity_df_article_ids, similarity_df_sim, similarity_levels):
    """
    Map each article_id to a tuple of similarity scores such that these scores correspond to the similarity scores of
    the article_ids if we would create the similarity_article_dict.
    :param similarity_df_article_ids: a dataframe containing the top-100 most similar articles for each article
    :param similarity_df_sim: a dataframe containing the similarity scores of the top-100 most similar articles for
    each article
    :param similarity_levels: a list (or tuple) of similarity levels
    :return: the resulting dictionary
    """
    original_article_ids = similarity_df_article_ids['article_index'].tolist()
    similar_article_sim = list(zip(*[similarity_df_sim[str(sim_level)].tolist() for sim_level in similarity_levels]))
    result = {original_article_ids[i]: similar_article_sim[i] for i in range(len(original_article_ids))}
    return result


def submission_odp(filename, creation=False):
    """
    Get the filename including the path to store/open a file containing a (part of a) submission file.
    :param filename: the filename of the file
    :param creation: if True, the directory is 'submission_creation' instead of 'submission'
    :return: the filename with path
    """
    directory = 'submission_creation' if creation else 'submission'
    create_directory_if_not_exists(directory)
    return odp(f'{directory}/{filename}')


def check_submission(filename, nr_customers):
    """
    Check submission file to avoid errors in Kaggle. To be checked:
    - Does the file has the right nr of lines?
    - Does it have the right column values?
    - Does each customer have 12 recommendations?
    :param filename: the name of the submission file
    :param nr_customers: the total numbr of customers in the dataset
    """
    submission = pd.read_csv(submission_odp(filename))
    print(f'Nr of customers in submission file: {submission["customer_id"].nunique()}')
    print(f'Nr of lines in submission file:     {submission.shape[0]}')
    print(f'Nr of customers:                    {nr_customers}')
    spaces = submission['prediction'].str.count('\s') + 1
    less_than_12_spaces = spaces[spaces < 12].shape[0]
    more_than_12_spaces = spaces[spaces > 12].shape[0]
    less_than_12_recommendations = less_than_12_spaces / spaces.shape[0]
    more_than_12_recommendations = more_than_12_spaces / spaces.shape[0]
    print(f'Customers with < 12 recommendations: {round(less_than_12_recommendations * 100, 2)}%')
    print(f'Customers with > 12 recommendations: {round(more_than_12_recommendations * 100, 2)}%')
    file = open(submission_odp(filename))
    first_line = file.readline().strip('\n')
    file.close()
    print(f'Nr of columns: {first_line.count(",") + 1} ({first_line})')
    print()
    print('Conclusion:')
    if submission.shape[0] != nr_customers:
        print(f'\t* Too {"many" if submission.shape[0] > nr_customers else "few"} lines')
    if submission["customer_id"].nunique() != nr_customers:
        print(f'\t* Too {"many" if submission.shape[0] > nr_customers else "few"} customers')
    if less_than_12_spaces > 0:
        print(f'\t* {less_than_12_spaces} customers have < 12 recommendations')
    elif more_than_12_spaces > 0:
        print(f'\t* {more_than_12_spaces} customers have > 12 recommendations')
    if first_line.split(',') != ['customer_id', 'prediction']:
        print(f'\t* Incorrect columns (expected [\'customer_id\', \'prediction\'], got {first_line.split(",")})')
    elif (submission.shape[0] == submission["customer_id"].nunique() == nr_customers) \
            and (more_than_12_spaces == less_than_12_spaces == 0):
        print('All fine!')
    return


def join_partial_submissions(base_file_name, trgt_file_name, nr_files):
    """
    Join a set of partial submission files into one submission file containing all customers
    :param base_file_name: base file name of the subfiles (will also be used to define the name of the final submission
    file)
    :param trgt_file_name: name of the target file
    :param nr_files: the nr of files to be joined
    """
    submission_parts = [
        pd.read_csv(submission_odp({base_file_name.replace("*", str(i + 1))}, creation=True))
        for i in range(nr_files)
    ]
    concatenation = pd.concat(submission_parts, ignore_index=True)
    concatenation.to_csv(submission_odp(trgt_file_name), index=False)
    return


def customer_show(history, prediction):
    """
    Create two image grids: one for the purchase history (the 12 most recently purchased items), the other for the
    recommendations
    :param history:
    :param prediction:
    :return:
    """
    figure_dict = {f'P{article_id}': load_img(f'{article_id[:3]}/{article_id}.jpg') for article_id in history}
    plot_img_grid(figure_dict, 3, 4, (6, 6))

    figure_dict = {f'R{article_id}': load_img(f'{article_id[:3]}/{article_id}.jpg') for article_id in prediction}
    plot_img_grid(figure_dict, 3, 4, (6, 6))

    return


def random_show_customer_case(filename, customer_transactions_df):
    """
    From a submission file, randomly show a customer case: show the 12 most recently purchased articles together with
    the recommendations made for this user.
    :param filename: the name of the submission file
    :param customer_transactions_df: the dataframe containing customer purchase history
    """
    submission = pd.read_csv(submission_odp(filename))
    customer = pd.DataFrame(customer_transactions_df.sample(1)).iloc[0]
    history = article_str_to_list(customer['purchase_history'])[-12:]
    prediction = article_str_to_list(
        submission[submission['customer_id'] == customer['customer_id']]['prediction'].iloc[0]
    )
    customer_show(history, prediction)
    return


def show_customer_case(customer_id, filename, customer_transactions_df):
    """
    From a submission file, show the customer case for the customer with customer_id: show the 12 most recently
    purchased articles together with the recommendations made for this user.
    :param customer_id: a customer_id
    :param filename: the name of the submission file
    :param customer_transactions_df: the dataframe containing customer purchase history
    """
    submission = pd.read_csv(submission_odp(filename))
    customer = pd.DataFrame(customer_transactions_df[customer_transactions_df['customer_id'] == customer_id]).iloc[0]
    history = article_str_to_list(customer['purchase_history'])[-12:]
    prediction = article_str_to_list(
        submission[submission['customer_id'] == customer_id]['prediction'].iloc[0]
    )
    customer_show(history, prediction)
    return
