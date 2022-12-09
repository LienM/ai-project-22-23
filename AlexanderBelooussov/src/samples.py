from recpack_samples import *
from similarity import get_similarity_samples, add_similarity

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None


def previous_week_customer_info(data):
    transactions = data['transactions']
    customers = data['customers']
    customers = customers[customers['customer_id'].isin(transactions['customer_id'])]

    # =====avg purchase price=====
    avg_purchase_price = transactions[['customer_id', 'week', 'price']]
    avg_purchase_price = avg_purchase_price.groupby(['customer_id', 'week'])['price'].sum().rename('price_sum').astype(
        'float32').reset_index()
    avg_purchase_price['purchase_count'] = \
        transactions[['customer_id', 'week', 'price']].groupby(['customer_id', 'week'])[
            'price'].count().to_numpy()
    avg_purchase_price['price_cumsum'] = avg_purchase_price.groupby('customer_id')['price_sum'].cumsum()
    avg_purchase_price['purchase_count_cumsum'] = avg_purchase_price.groupby('customer_id')['purchase_count'].cumsum()
    avg_purchase_price['avg_purchase_price'] = (
            avg_purchase_price['price_cumsum'] / avg_purchase_price['purchase_count_cumsum']).astype('float32')
    avg_purchase_price.drop(['price_sum', 'purchase_count', 'price_cumsum', 'purchase_count_cumsum'], axis=1,
                            inplace=True)
    avg_purchase_price['week'] += 1
    customer_week_info = avg_purchase_price.set_index(['customer_id', 'week']).unstack()
    customer_week_info.fillna(method='ffill', inplace=True, axis=1)
    customer_week_info = pdc.downcast(customer_week_info.stack().reset_index())

    # =====weeks since last purchase=====
    purchase_weeks = transactions[['customer_id', 'week']].drop_duplicates()
    purchase_weeks['has_purchased'] = int(1)
    customer_week_info = merge_downcast(customer_week_info, purchase_weeks, on=['customer_id', 'week'], how='left')
    customer_week_info['has_purchased'].fillna(int(0), inplace=True)
    # https://stackoverflow.com/a/44421059
    mask = customer_week_info.groupby('customer_id').has_purchased.cumsum().astype(bool)  # Mask starting zeros as NaN
    customer_week_info = customer_week_info.assign(since_last_purchase=customer_week_info.groupby(
        customer_week_info.has_purchased.astype(bool).cumsum()).cumcount().where(mask))
    # shift column down by 1
    customer_week_info['since_last_purchase'] = customer_week_info.groupby('customer_id')['since_last_purchase'].shift(
        1)
    customer_week_info['since_last_purchase'] += 1
    customer_week_info['since_last_purchase'].fillna(-1, inplace=True)
    customer_week_info['since_last_purchase'] = customer_week_info['since_last_purchase'].astype('int8')
    customer_week_info.drop('has_purchased', axis=1, inplace=True)

    return customer_week_info


def previous_week_article_info(data):
    """
    Data about sales in previous week(s) for each article and week
    :param data:
    :return:
    """
    transactions = data['transactions']

    # cross join of article ids and weeks
    weeks = transactions.week.unique()
    article_ids = transactions.article_id.unique()
    weeks, article_ids = np.meshgrid(weeks, article_ids)
    weeks = weeks.flatten()
    article_ids = article_ids.flatten()
    article_week_info = pd.DataFrame({'week': weeks, 'article_id': article_ids})
    customers = data['customers']

    # =====bestseller rank=====
    mean_price = transactions \
        .groupby(['week', 'article_id'])['price'].mean()

    weekly_sales_rank = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='min', ascending=False) \
        .rename('bestseller_rank').astype('int16')

    previous_week_rank = merge_downcast(weekly_sales_rank, mean_price, on=['week', 'article_id']).reset_index()
    # previous_week_rank.week += 1
    article_week_info = merge_downcast(article_week_info, previous_week_rank, on=['week', 'article_id'], how='left')
    article_week_info.bestseller_rank.fillna(article_week_info.bestseller_rank.max() + 100, inplace=True)
    article_week_info['bestseller_rank'] = pd.to_numeric(article_week_info['bestseller_rank'], downcast='integer')
    article_week_info["price"] = article_week_info.groupby("article_id")["price"].transform(
        lambda x: x.fillna(x.mean()))

    # =====sales in last week=====
    weekly_sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .rename('1w_sales').astype('int16').reset_index()
    # weekly_sales.week += 1

    article_week_info = pd.merge(article_week_info, weekly_sales, on=['week', 'article_id'], how='left')
    article_week_info['1w_sales'].fillna(0, inplace=True)
    article_week_info['1w_sales'] = pd.to_numeric(article_week_info['1w_sales'], downcast='integer')
    article_week_info = pdc.downcast(article_week_info)

    # =====mean buyer age=====

    mean_buyer_age = merge_downcast(transactions[['customer_id', 'article_id', 'week']],
                                    customers[['customer_id', 'age']],
                                    on='customer_id', how='left')
    mean_buyer_age = mean_buyer_age.groupby(['article_id', 'week'])['age'].sum().rename('age_sum').astype(
        'int16').reset_index()
    mean_buyer_age['purchase_count'] = \
        transactions[['customer_id', 'article_id', 'week']].groupby(['article_id', 'week'])[
            'customer_id'].count().to_numpy()

    article_week_info = merge_downcast(article_week_info, mean_buyer_age, on=['article_id', 'week'], how='left')
    article_week_info['age_sum'].fillna(0, inplace=True)
    article_week_info['age_sum'] = pd.to_numeric(article_week_info['age_sum'], downcast='integer')
    article_week_info['purchase_count'].fillna(0, inplace=True)
    article_week_info['purchase_count'] = pd.to_numeric(article_week_info['purchase_count'], downcast='integer')
    article_week_info['age_cumsum'] = article_week_info.groupby('article_id')['age_sum'].cumsum()
    article_week_info['purchase_count_cumsum'] = article_week_info.groupby('article_id')['purchase_count'].cumsum()
    article_week_info['purchase_count_cumsum'] += 1
    article_week_info['mean_buyer_age'] = article_week_info['age_cumsum'] / article_week_info['purchase_count_cumsum']
    article_week_info['mean_buyer_age'] = article_week_info['mean_buyer_age'].astype('int16')
    article_week_info.drop(['age_sum', 'purchase_count', 'age_cumsum', 'purchase_count_cumsum'], axis=1, inplace=True)

    # =====sales in last 4 weeks=====
    article_week_info['4w_sales'] = article_week_info.groupby('article_id')['1w_sales'].rolling(4,
                                                                                                min_periods=1).sum().reset_index(
        0, drop=True)
    article_week_info['4w_sales'] = pd.to_numeric(article_week_info['4w_sales'], downcast='integer')
    # =====sales in all last weeks=====
    article_week_info['all_sales'] = article_week_info.groupby('article_id')['1w_sales'].cumsum()
    article_week_info['all_sales'] = pd.to_numeric(article_week_info['all_sales'], downcast='integer')
    # =====bestseller rank in last 4 weeks=====
    bestseller_4w = article_week_info.groupby('week')['4w_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_4w').astype('int16').reset_index()
    article_week_info['bestseller_4w'] = bestseller_4w['bestseller_4w']
    # =====bestseller rank in all last weeks=====
    bestseller_all = article_week_info.groupby('week')['all_sales'] \
        .rank(method='dense', ascending=False).rename('bestseller_all').astype('int16').reset_index()
    article_week_info['bestseller_all'] = bestseller_all['bestseller_all']

    article_week_info['week'] += 1  # shift week so that it is about the previous week
    # print(article_week_info.head(100))
    return article_week_info


def samples(data, n_train_weeks=12, n=12, ratio=1, verbose=True, methods=None, scale_n=True):
    """
    Generate samples (positive, negative, candidates)
    :param data:
    :param n_train_weeks:
    :param n: Number of samples per method, higher = higher recall
    :return:
    """
    if methods is None:
        methods = ['itemknn', 'w2v', 'l0']
    bestseller_types = ['bestseller_rank', 'bestseller_4w', 'bestseller_all']

    # ========================================================
    # limit scope to train weeks
    transactions = data['transactions']
    test_week = transactions.week.max() + 1
    if n_train_weeks > 0:
        transactions = transactions[transactions.week > transactions.week.max() - n_train_weeks]
        data['transactions'] = transactions
    # ========================================================
    # gather info about previous weeks
    previous_week_info = previous_week_article_info(data)
    previous_week_cust_info = previous_week_customer_info(data)
    data['article_week_info'] = previous_week_info
    data['purchase_history'] = make_purchase_history(data['transactions'])

    # ========================================================
    # repurchasing
    c2weeks = transactions.groupby('customer_id')['week'].unique()
    c2weeks2shifted_weeks = {}
    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0] - 1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

    candidates_last_purchase = transactions.copy()

    weeks = []
    for i, (c_id, week) in enumerate(zip(transactions['customer_id'], transactions['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])

    candidates_last_purchase.week = weeks
    # ========================================================

    # get all customers that made a purchase and the week of their last purchase
    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()
    # get all customers that made a purchase
    test_set_transactions = unique_transactions.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_transactions.week = test_week
    bestseller_candidates = pd.DataFrame()
    for bestseller_type in bestseller_types:
        # for customers who made purchases, generate negative samples based on the popular items at the time
        # get the most popular items from the previous week
        bestsellers_previous_week = previous_week_info[previous_week_info[bestseller_type] <= n] \
            [['week', 'article_id', 'bestseller_rank', 'price']].sort_values(['week', 'bestseller_rank'])

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

    # ========================================================
    recpack_candidates = pd.DataFrame()
    recpack_methods = []
    if 'itemknn' in methods:
        recpack_methods.append('itemknn')
    if 'p2v' in methods:
        recpack_methods.append('prod2vec')
    for method in recpack_methods:
        if verbose:
            print(f"Generating samples with RecPack: {method}")
        # some initial recpack candidates
        for w in tqdm(range(transactions.week.min() + 1, transactions.week.max() + 2),
                      desc=f"Recpack samples per week"):
            if scale_n:
                i = int((w - transactions.week.min()) / (transactions.week.max() + 1 - transactions.week.min()) * n)
                i = max(1, i)
            else:
                i = n
            customers_to_use = unique_transactions[
                unique_transactions.week == w] if w < transactions.week.max() + 1 else test_set_transactions
            transactions_w = transactions[transactions.week < w]
            recpack = get_recpack_samples(transactions_w, i, algorithm=method,
                                          customers=customers_to_use.customer_id.unique())
            recpack['week'] = w
            recpack_samples = pd.merge(
                customers_to_use,
                recpack,
                on=['week', 'customer_id']
            )
            recpack_candidates = concat_downcast([recpack_candidates, recpack_samples])
            recpack_candidates.drop_duplicates(['week', 'customer_id', 'article_id'], inplace=True)

    # add price
    if len(recpack_methods):
        recpack_candidates = merge_downcast(recpack_candidates, previous_week_info[['week', 'article_id', 'price']],
                                            on=['week', 'article_id'], how='left')
        recpack_candidates.drop_duplicates(['week', 'customer_id', 'article_id'], inplace=True)

    # ========================================================
    # similarity candidates
    similarity_candidates = pd.DataFrame()
    types = []
    if 'w2v' in methods: types.append('w2v')
    if 'l0' in methods: types.append('l0')
    for sim_type in types:
        sim_matrix, article_map, candidates = get_similarity_samples(data, transactions, n, sim_type,
                                                                     unique_transactions, test_set_transactions,
                                                                     scale_n)
        data[f'{sim_type}_similarity'] = sim_matrix
        data[f'{sim_type}_similarity_index'] = article_map

        similarity_candidates = concat_downcast([similarity_candidates, candidates])
    # ========================================================
    # set purchased for positive samples
    transactions['purchased'] = 1
    # combine transactions and candidates
    samples = concat_downcast([transactions,
                               candidates_last_purchase,
                               bestseller_candidates,
                               recpack_candidates,
                               similarity_candidates
                               ])
    # set purchased to 0 for candidates and negative samples
    samples.purchased.fillna(0, inplace=True)
    samples.purchased = samples.purchased.astype(np.int8)
    # only keep the first occurrence of a customer-week-article combination
    # this will keep bought articles since they are concatenated first
    samples.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)

    # remove some training samples to match the ratio of positive to negative samples
    train = samples[samples.week != test_week].sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    candidates = samples[samples.week == test_week].sort_values(by=['week', 'customer_id']).reset_index(drop=True)
    pos_samples = train[train.purchased == 1]
    n_pos_samples = pos_samples.shape[0]
    neg_samples = train[train.purchased == 0]
    n_neg_samples = min(int(n_pos_samples * ratio), neg_samples.shape[0])
    if n_neg_samples < neg_samples.shape[0]:
        neg_sample = train[train.purchased == 0].sample(n=n_neg_samples, random_state=42)
        train = concat_downcast([pos_samples, neg_sample])
        samples = concat_downcast([train, candidates])

    # print some statistics about our samples
    p_mean = n_pos_samples / (n_pos_samples + n_neg_samples)
    n_candidates = samples.loc[samples['week'] == test_week].shape[0]
    n_customers_c = candidates.customer_id.nunique()
    print(f"REMAINING RATIO OF POSITIVE SAMPLES: {p_mean * 100:.2f}%, {1}:{(1 - p_mean) / p_mean:.2f}")
    print(f"REMAINING #CANDIDATES: {n_candidates}, #SAMPLES: {n_pos_samples + n_neg_samples}")
    print(
        f"#CUSTOMERS WITH CANDIDATES: {n_customers_c}, {n_customers_c / data['customers'].shape[0] * 100:.2f}% OF ALL CUSTOMERS")

    # add info about sales in previous weeks
    samples = merge_downcast(
        samples,
        previous_week_info[['week', 'article_id',
                            'bestseller_rank', 'bestseller_4w', 'bestseller_all',
                            '1w_sales', '4w_sales', 'all_sales',
                            'mean_buyer_age']],
        on=['week', 'article_id'],
        how='left'
    )
    samples = merge_downcast(
        samples,
        previous_week_cust_info,
        on=['customer_id', 'week'],
        how='left'
    )
    samples['since_last_purchase'].fillna(-1, inplace=True)
    # ========================================================
    # finalise samples
    samples = samples[samples.week != samples.week.min()]  # remove first week due to lack of information

    # add info about w2v similarity
    if 'itemknn' in methods:
        samples = add_recpack_score(samples, transactions, algorithm='itemknn')
    if 'p2v' in methods:
        samples = add_recpack_score(samples, transactions, algorithm='prod2vec')
    if 'w2v' in methods:
        samples['w2v_sim'] = add_similarity(samples, data['purchase_history'], data['w2v_similarity'],
                                            data['w2v_similarity_index'])
        del data['w2v_similarity'], data['w2v_similarity_index']
    if 'l0' in methods:
        samples['l0_sim'] = add_similarity(samples, data['purchase_history'], data['l0_similarity'],
                                           data['l0_similarity_index'])
        del data['l0_similarity'], data['l0_similarity_index']

    # merge but ignore w2v_i columns
    columns = [c for c in data['articles'].columns if 'w2v' not in c]
    samples = merge_downcast(samples, data['articles'][columns], on='article_id', how='left')  # merge article info
    samples = merge_downcast(samples, data['customers'], on='customer_id', how='left')  # merge customer info

    # add differences
    samples['age'] = samples['age'].astype('int8')
    samples['mean_buyer_age'] = samples['mean_buyer_age'].astype('int8')
    samples['age_diff'] = (samples['age'] - samples['mean_buyer_age'])
    samples['age_diff_abs'] = samples['age_diff'].abs()
    samples['price_diff'] = (samples['price'] - samples['avg_purchase_price'])
    samples['price_diff_abs'] = samples['price_diff'].abs()

    if verbose: print(samples.info())

    samples.sort_values(['week', 'customer_id'], inplace=True)
    samples.reset_index(drop=True, inplace=True)
    data['samples'] = samples
    if verbose:
        print(samples.head(200).sort_values(by=['article_id', 'week'], inplace=False))
        print(f"Samples shape: {samples.shape}")
        print(samples[samples.week == test_week].head(200).sort_values(by=['article_id', 'week'], inplace=False))

    data['test_week'] = test_week

    return data
