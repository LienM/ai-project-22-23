import cv2
import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd

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
    if not os.path.isdir(OUTPUT_DATA_PATH):
        os.mkdir(OUTPUT_DATA_PATH)
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


def remove_output_directory_sequence_if_exists(base_directory):
    """
    Remove the directory sequence pattern with given base name from the output path.
    All directories that start with base_directory are removed
    :param base_directory: a base directory
    """
    content = os.listdir(odp(filename=''))
    for directory in content:
        if directory.startswith(base_directory):
            shutil.rmtree(odp(filename=directory))


def remove_file_if_exists(filename):
    """
    Remove the file with given name from the output path
    :param filename: name of the file to remove
    """
    if os.path.isfile(odp(filename=filename)):
        shutil.rmtree(odp(filename=filename))


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


def load_img(filename, for_display=False):
    """
    Load the image from the file with given name
    :param filename: filename of the image
    :param for_display: load image for display (in that case return the full size image
    :return: the image
    """
    img = cv2.imread(idp(f'images{"_full_size" if for_display else ""}/{filename}'))
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
    for ind, title in enumerate(fig_dict):
        try:
            ax_list.ravel()[ind].imshow(cv2.cvtColor(fig_dict[title], cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(e)
            pass
        ax_list.ravel()[ind].set_title(title)
    for ind in range(n_rows * n_cols):
        ax_list.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional
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


def check_submission(filename):
    """
    Check submission file to avoid errors in Kaggle. To be checked:
    - Does the file has the right nr of lines?
    - Does it have the right column values?
    - Does each customer have 12 recommendations?
    :param filename: the name of the submission file
    """
    customer_df = pd.read_feather(idp('customer_transactions_processed.feather'))
    submission = pd.read_csv(filename)
    print(f'Nr of customers in submission file: {submission["customer_id"].nunique()}')
    print(f'Nr of lines in submission file:     {submission.shape[0]}')
    print(f'Nr of customers:                    {customer_df.shape[0]}')
    print(f'All customers included:             '
          f'{sorted(submission["customer_id"].values.tolist()) == sorted(customer_df["customer_id"].values.tolist())}')
    spaces = submission['prediction'].str.count('\s') + 1
    less_than_12_spaces = spaces[spaces < 12].shape[0]
    more_than_12_spaces = spaces[spaces > 12].shape[0]
    less_than_12_recommendations = less_than_12_spaces / spaces.shape[0]
    more_than_12_recommendations = more_than_12_spaces / spaces.shape[0]
    print(f'Customers with < 12 recommendations: {round(less_than_12_recommendations * 100, 2)}%')
    print(f'Customers with > 12 recommendations: {round(more_than_12_recommendations * 100, 2)}%')
    file = open(filename)
    first_line = file.readline().strip('\n')
    file.close()
    print(f'Nr of columns: {first_line.count(",") + 1} ({first_line})')
    print()
    print('Conclusion:')
    if submission.shape[0] != customer_df.shape[0]:
        print(f'\t* Too {"many" if submission.shape[0] > customer_df.shape[0] else "few"} lines')
    if submission["customer_id"].nunique() != customer_df.shape[0]:
        print(f'\t* Too {"many" if submission.shape[0] > customer_df.shape[0] else "few"} customers')
    if less_than_12_spaces > 0:
        print(f'\t* {less_than_12_spaces} customers have < 12 recommendations')
    elif more_than_12_spaces > 0:
        print(f'\t* {more_than_12_spaces} customers have > 12 recommendations')
    if first_line.split(',') != ['customer_id', 'prediction']:
        print(f'\t* Incorrect columns (expected [\'customer_id\', \'prediction\'], got {first_line.split(",")})')
    elif (submission.shape[0] == submission["customer_id"].nunique() == customer_df.shape[0]) \
            and (more_than_12_spaces == less_than_12_spaces == 0):
        print('All fine!')
        return True
    return False


def check_intermediate_article_result(directory):
    """
    Check for a given directory whether all files have as much lines as there are articles.
    Goal: check whether joining files in batch_process.py join_files led to the right amount of output
    :param directory: the name of the directory to check
    """
    nr_articles = pd.read_feather(idp('articles_processed.feather')).shape[0]
    sub_directories = list_directory_if_exists(odp(directory))
    for sub_directory in sub_directories:
        files = list_directory_if_exists(odp(f'{directory}/{sub_directory}'))
        for file in files:
            if file.endswith('.csv'):
                content = pd.read_csv(odp(f'{directory}/{sub_directory}/{file}'))
            elif file.endswith('.feather'):
                content = pd.read_feather(odp(f'{directory}/{sub_directory}/{file}'))
            else:
                continue
            if content.shape[0] == nr_articles:
                print('OK', f'{directory}/{sub_directory}/{file}')
            else:
                print('!Not ok!', f'{directory}/{sub_directory}/{file}',
                      '--> expected', nr_articles, 'got', content.shape[0])


def customer_show(history, prediction):
    """
    Create two image grids: one for the purchase history (the 12 most recently purchased items), the other for the
    recommendations
    :param history:
    :param prediction:
    :return:
    """
    alphabet = 'abcdefghijkl' # for figure order (important if pairwise)
    if len(history) > 0:
        figure_dict = {
            f'{alphabet[i]}.P{article_id}': load_img(f'{article_id[:3]}/{article_id}.jpg', True)
            for i, article_id in enumerate(history)
        }
        plot_img_grid(figure_dict, 3, 4, (6, 6))

    figure_dict = {
        f'{alphabet[i]}.R{article_id}': load_img(f'{article_id[:3]}/{article_id}.jpg', True)
        for i, article_id in enumerate(prediction)
    }
    plot_img_grid(figure_dict, 3, 4, (6, 6))

    return


def random_show_customer_case(filename, customer_transactions_df):
    """
    From a submission file, randomly show a customer case: show the 12 most recently purchased articles together with
    the recommendations made for this user.
    :param filename: the name of the submission file
    :param customer_transactions_df: the dataframe containing customer purchase history
    """
    submission = pd.read_csv(filename)
    customer = pd.DataFrame(customer_transactions_df.sample(1)).iloc[0]
    history = article_str_to_list(customer['purchase_history'])[:12]
    prediction = article_str_to_list(
        submission[submission['customer_id'] == customer['customer_id']]['prediction'].iloc[0]
    )
    customer_show(history, prediction)
    return


def show_specific_customer_case(customer_id, filename, customer_transactions_df, with_history=True):
    """
    From a submission file, show the customer case for the customer with customer_id: show the 12 most recently
    purchased articles together with the recommendations made for this user.
    :param customer_id: a customer_id or an index of a customer in the customer transactions dataframe
    :param filename: the name of the submission file
    :param customer_transactions_df: the dataframe containing customer purchase history
    :param with_history: a boolean indicating whether only the recommendations should be displayed, or also the history
    """
    if isinstance(customer_id, int):
        # the index is given, now get the id
        customer_id = customer_transactions_df.iloc[customer_id]['customer_id']
    submission = pd.read_csv(filename)
    customer = pd.DataFrame(customer_transactions_df[customer_transactions_df['customer_id'] == customer_id]).iloc[0]
    if with_history:
        history = article_str_to_list(customer['purchase_history'])[:12]
    else:
        history = []
    prediction = article_str_to_list(
        submission[submission['customer_id'] == customer_id]['prediction'].iloc[0]
    )
    customer_show(history, prediction)
    return


def current_time_stamp():
    return time.strftime("%H:%M:%S", time.localtime())
