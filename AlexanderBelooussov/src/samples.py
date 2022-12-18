import warnings

from historical_info import previous_week_customer_info, previous_week_article_info
from recpack_samples import *
from recpack_samples import generate_recpack_samples
from similarity import add_similarity, generate_similarity_candidates
from random_samples import generate_random_samples

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None


def samples(data_dict, n_train_weeks=12, n=12, ratio=1, verbose=True, methods=None, scale_n=False):
    """
    Generate samples (positive, negative, candidates)
    :param data_dict:
    :param n_train_weeks:
    :param n: Number of samples per method, higher = higher recall
    :param ratio: Int, amount of negative samples for each positive sample
    :param verbose: Bool, whether to output more info
    :param methods: list containing used methods, subset of ['l0', 'w2v', 'itemknn', 'p2v', 'random']
    :param scale_n: Whether to reduce amount of negative samples for earlier weeks
    :return:
    """
    if methods is None:
        methods = ['itemknn', 'w2v', 'l0']
    bestseller_types = ['bestseller_rank', 'bestseller_4w', 'bestseller_all']

    # ========================================================
    # limit scope to train weeks
    transactions_train = data_dict['transactions']
    test_week = transactions_train.week.max() + 1

    # ========================================================
    # gather info about previous weeks
    if n_train_weeks > 0:
        transactions_train = transactions_train[transactions_train.week >
                                                transactions_train.week.max() - n_train_weeks * 2]
        data_dict['transactions'] = transactions_train
    previous_week_info = previous_week_article_info(data_dict)
    previous_week_cust_info = previous_week_customer_info(data_dict)
    data_dict['article_week_info'] = previous_week_info

    data_dict['purchase_history'] = make_purchase_history(data_dict['transactions'])
    # ========================================================
    # Repurchasing candidates
    candidates_last_purchase = generate_repurchase_candidates(test_week, transactions_train)

    # ========================================================
    # get all customers that made a purchase and the week of their last purchase
    unique_transactions = transactions_train \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()
    # get all customers that made a purchase
    test_set_transactions = unique_transactions.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_transactions.week = test_week

    # ========================================================
    # Popularity candidates
    bestseller_candidates = pd.DataFrame()
    bestseller_candidates = generate_bestseller_samples(bestseller_candidates, bestseller_types, n, previous_week_info,
                                                        test_set_transactions, unique_transactions)

    # ========================================================
    # RecPack samples
    recpack_candidates = pd.DataFrame()
    recpack_methods = []
    if 'itemknn' in methods:
        recpack_methods.append('itemknn')
    if 'p2v' in methods:
        recpack_methods.append('prod2vec')
    recpack_candidates = generate_recpack_samples(n, n_train_weeks, previous_week_info, recpack_candidates,
                                                  recpack_methods, scale_n, test_set_transactions, transactions_train,
                                                  unique_transactions, verbose)

    # ========================================================
    # similarity candidates
    similarity_candidates = pd.DataFrame()
    types = []
    if 'w2v' in methods:
        types.append('w2v')
    if 'l0' in methods:
        types.append('l0')
    similarity_candidates = generate_similarity_candidates(data_dict, n, n_train_weeks, scale_n, similarity_candidates,
                                                           test_set_transactions, transactions_train, types,
                                                           unique_transactions)

    # ========================================================
    # Random candidates
    random_samples = pd.DataFrame()
    if "random" in methods:
        random_samples = generate_random_samples(transactions_train, n, n_train_weeks)
        random_samples = merge_downcast(random_samples,
                                        previous_week_info[['week', 'article_id', 'price']],
                                        on=['week', 'article_id'], how='left')

    # ========================================================
    # finalise samples

    # set purchased for positive samples
    transactions_train['purchased'] = 1
    # combine transactions and candidates
    to_concat = [candidates_last_purchase.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True),
                 bestseller_candidates,
                 recpack_candidates,
                 similarity_candidates,
                 random_samples]

    all_samples = to_concat[0]
    descriptions = ["Repurchase", "Popularity", "RecPack", "Similarity", "Random"]
    for i in range(1, len(to_concat)):
        second = to_concat[i]
        if len(second) == 0:
            continue
        all_samples = concat_and_remove_dupes(all_samples, second, verbose, descriptions[i])
    all_samples = concat_and_remove_dupes(transactions_train, all_samples, verbose, "All")

    # set purchased to 0 for candidates and negative samples
    all_samples.purchased.fillna(0, inplace=True)
    all_samples.purchased = all_samples.purchased.astype(np.int8)

    if n_train_weeks > 0:
        # only keep n_train_weeks weeks for the purpose of training the ranker
        all_samples = all_samples[all_samples.week >= transactions_train.week.max() - n_train_weeks]
    else:
        # remove first week due to lack of information
        # only needed if using all weeks
        all_samples = all_samples[all_samples.week != all_samples.week.min()]

    # remove some training samples to match the ratio of positive to negative samples
    all_samples = downsample_negative_samples(all_samples, data_dict, ratio, test_week, verbose)

    # add info about sales in previous weeks
    all_samples = merge_downcast(
        all_samples,
        previous_week_info[['week', 'article_id',
                            'bestseller_rank', 'bestseller_4w', 'bestseller_all',
                            '1w_sales', '4w_sales', 'all_sales',
                            'mean_buyer_age']],
        on=['week', 'article_id'],
        how='left'
    )
    all_samples = merge_downcast(
        all_samples,
        previous_week_cust_info,
        on=['customer_id', 'week'],
        how='left'
    )
    all_samples['since_last_purchase'].fillna(-1, inplace=True)
    all_samples['avg_purchase_price'].fillna(-1, inplace=True)

    # add info about w2v similarity
    if 'itemknn' in methods:
        all_samples = add_recpack_score(all_samples, transactions_train, algorithm='itemknn')
    if 'p2v' in methods:
        all_samples = add_recpack_score(all_samples, transactions_train, algorithm='prod2vec')
    if 'w2v' in methods:
        all_samples['w2v_sim'] = add_similarity(all_samples, data_dict['purchase_history'], data_dict['w2v_similarity'],
                                                data_dict['w2v_similarity_index'])
        del data_dict['w2v_similarity'], data_dict['w2v_similarity_index']
    if 'l0' in methods:
        all_samples['l0_sim'] = add_similarity(all_samples, data_dict['purchase_history'], data_dict['l0_similarity'],
                                               data_dict['l0_similarity_index'])
        del data_dict['l0_similarity'], data_dict['l0_similarity_index']

    # merge with static data but ignore w2v_i columns
    columns = [c for c in data_dict['articles'].columns if 'w2v' not in c]
    all_samples = merge_downcast(all_samples, data_dict['articles'][columns], on='article_id',
                                 how='left')  # merge article info
    all_samples = merge_downcast(all_samples, data_dict['customers'], on='customer_id',
                                 how='left')  # merge customer info

    # add differences
    all_samples['age'] = all_samples['age'].astype('int8')
    all_samples['mean_buyer_age'] = all_samples['mean_buyer_age'].astype('int8')
    all_samples['age_diff'] = (all_samples['age'] - all_samples['mean_buyer_age'])
    all_samples['age_diff_abs'] = all_samples['age_diff'].abs()
    all_samples['price_diff'] = (all_samples['price'] - all_samples['avg_purchase_price'])
    all_samples['price_diff_abs'] = all_samples['price_diff'].abs()

    if verbose:
        print(all_samples.info())

    all_samples.sort_values(['week', 'customer_id'], inplace=True)
    all_samples.reset_index(drop=True, inplace=True)
    data_dict['samples'] = all_samples
    if verbose:
        print(all_samples.head(200).sort_values(by=['article_id', 'week'], inplace=False))
        print(f"Samples shape: {all_samples.shape}")
        print(
            all_samples[all_samples.week == test_week].head(200).sort_values(by=['article_id', 'week'], inplace=False))

    data_dict['test_week'] = test_week

    return data_dict


def downsample_negative_samples(all_samples, data_dict, ratio, test_week, verbose):
    train = all_samples[all_samples.week != test_week].sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    candidates = all_samples[all_samples.week == test_week] \
        .sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    pos_samples = train[train.purchased == 1]
    n_pos_samples = pos_samples.shape[0]
    neg_samples = train[train.purchased == 0]
    n_neg_samples = min(int(n_pos_samples * ratio), neg_samples.shape[0])
    if n_neg_samples < neg_samples.shape[0]:
        neg_sample = train[train.purchased == 0].sample(n=n_neg_samples, random_state=42)
        train = concat_downcast([pos_samples, neg_sample])
        all_samples = concat_downcast([train, candidates])
    if verbose:
        # print some statistics about our samples
        p_mean = n_pos_samples / (n_pos_samples + n_neg_samples)
        n_candidates = all_samples.loc[all_samples['week'] == test_week].shape[0]
        n_customers_c = candidates.customer_id.nunique()
        print(f"REMAINING RATIO OF POSITIVE SAMPLES: {p_mean * 100:.2f}%, {1}:{(1 - p_mean) / p_mean:.2f}")
        print(f"REMAINING #CANDIDATES: {n_candidates}, #SAMPLES: {n_pos_samples + n_neg_samples}")
        print(
            f"#CUSTOMERS WITH CANDIDATES: {n_customers_c}, "
            f"{n_customers_c / data_dict['customers'].shape[0] * 100:.2f}% OF ALL CUSTOMERS")
    return all_samples


def generate_bestseller_samples(bestseller_candidates, bestseller_types, n, previous_week_info, test_set_transactions,
                                unique_transactions):
    print(f"Generating popularity samples", end="")
    for bestseller_type in bestseller_types:
        # for customers who made purchases, generate negative samples based on the popular items at the time
        # get the most popular items from the previous week
        bestsellers_previous_week = previous_week_info[previous_week_info[bestseller_type] <= n][
            ['week', 'article_id', 'bestseller_rank', 'price']].sort_values(['week', 'bestseller_rank'])

        # join popular items and customers + purchase weeks
        candidates_bestsellers = merge_downcast(
            unique_transactions,
            bestsellers_previous_week,
            on='week',
        )
        # candidates_bestsellers = pdc.downcast(candidates_bestsellers)

        # join popular items and customers
        candidates_bestsellers_test_week = merge_downcast(
            test_set_transactions,
            bestsellers_previous_week,
            on='week'
        )
        # candidates_bestsellers_test_week = pdc.downcast(candidates_bestsellers_test_week)
        bestseller_candidates = concat_downcast([bestseller_candidates,
                                                 candidates_bestsellers,
                                                 candidates_bestsellers_test_week])
    # drop rank, to be added later
    bestseller_candidates.drop(columns='bestseller_rank', inplace=True)
    print("\r", end="")
    return bestseller_candidates


def generate_repurchase_candidates(test_week, transactions_train):
    # repurchasing
    print(f"Adding repurchasing candidates", end="")
    c2weeks = transactions_train.groupby('customer_id')['week'].unique()
    c2weeks2shifted_weeks = {}
    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week
    candidates_last_purchase = transactions_train.copy()
    weeks = []
    for i, (c_id, week) in enumerate(zip(transactions_train['customer_id'], transactions_train['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])
    candidates_last_purchase.week = weeks
    print(f"\r", end="")
    return candidates_last_purchase


def concat_and_remove_dupes(first, second, verbose=True, kind="New"):
    new_samples = len(second)
    first = concat_downcast([first, second])
    duplicates = first[['customer_id', 'article_id', 'week']].duplicated(keep="first").sum()
    if verbose:
        print(f"{kind} samples added, {duplicates / new_samples * 100:.2f}% are duplicates")
    if duplicates / new_samples > 0.25:
        warnings.warn(f"Found {duplicates} duplicates from {new_samples} "
                      f"{kind} samples ({duplicates / new_samples * 100:.2f}%)")
    first.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)
    return first
