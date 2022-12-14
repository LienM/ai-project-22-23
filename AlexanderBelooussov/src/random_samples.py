from utils import merge_downcast


def generate_random_samples(transactions, n, n_train_weeks):
    """
    Generate random samples of size n from transactions.
    :param transactions: DataFrame with transactions
    :param n: number of samples to generate (total will be: n * #transactions in n_train_weeks)
    :param n_train_weeks: number of weeks to use for training
    :return: DataFrame with samples
    """
    transactions = transactions[transactions.week > transactions.week.max() - n_train_weeks]
    # calculate amount of needed samples
    n_samples = n * len(transactions)
    # generate random customers
    customers = transactions['customer_id'].sample(n=n_samples, replace=True).reset_index(drop=True)
    # generate random items that were bought previously
    items = transactions['article_id'].sample(n=n_samples, replace=True).reset_index(drop=True)
    # generate random week values
    weeks = transactions['week'].sample(n=n_samples, replace=True).reset_index(drop=True)
    samples = merge_downcast(customers, items, left_index=True, right_index=True)
    samples = merge_downcast(samples, weeks, left_index=True, right_index=True)
    return samples
