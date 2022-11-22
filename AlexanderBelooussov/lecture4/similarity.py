from multiprocessing import Pool

from utils import *
from sklearn.metrics.pairwise import cosine_similarity


def l0_similarity(articles, transactions):
    """
    Get similarity matrix based on L0 distance
    :param articles:
    :param transactions:
    :return: similarity matrix, article ids found in the matrix, dict mapping article ids to matrix indices
    """
    cols = ["product_code", "prod_name", "product_type_no", "product_group_name", "graphical_appearance_no",
            "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id", "index_code",
            "index_group_no", "department_no", "section_no", "garment_group_no"]
    # filter articles with no sales
    articles = articles[articles.article_id.isin(transactions.article_id.unique())]
    article_ids = articles.article_id.to_numpy()
    article_map = {article_ids[i]: i for i in range(len(article_ids))}

    # get category vectors
    vecs = articles[cols].to_numpy()
    # get similarity matrix
    sim = np.zeros((len(articles), len(articles)))
    for i, row in enumerate(vecs):
        distances = np.linalg.norm(vecs - row, axis=1, ord=0)
        sim_vec = 1 - distances / len(cols)
        sim[i] = np.array(sim_vec, dtype=np.float32)

    return sim, article_ids, article_map


def w2v_similarity(articles, transactions):
    """
    Get similarity matrix based on word2vec
    :param articles:
    :param transactions:
    :return: similarity matrix, article ids found in the matrix, dict mapping article ids to matrix indices
    """
    cols = articles.columns
    # filter articles with no sales
    articles = articles[articles.article_id.isin(transactions.article_id.unique())]
    article_ids = articles.article_id.to_numpy()
    article_map = {article_ids[i]: i for i in range(len(article_ids))}
    # get w2v vectors for each article
    w2v = articles[[c for c in cols if c.startswith('w2v_')]].to_numpy()
    # get similarity matrix
    sim = cosine_similarity(w2v)
    return sim, article_ids, article_map


# def get_similar_items_customer(customer, sim, article_ids, purchase_history, n):
#     return pd.DataFrame({'customer_id': customer,
#                          'article_id': article_ids[np.argsort(sim[purchase_history[customer]].mean(axis=0))[-n:]],
#                          })


def get_similar_items(articles, transactions, n, sim_type, last_week_customers_only=False):
    """
    Get top n similar items for each user (with a purchase history)
    :param articles:
    :param transactions:
    :param n:
    :param sim_type:
    :param last_week_customers_only:
    :return: similarity matrix, dict mapping article ids to matrix indices, dataframe with similar items
    """
    if sim_type == "l0":
        sim, article_ids, article_map = l0_similarity(articles, transactions)
    elif sim_type == "w2v":
        sim, article_ids, article_map = w2v_similarity(articles, transactions)
    else:
        raise ValueError("Invalid similarity type")

    if last_week_customers_only:
        customers = transactions[transactions.week == transactions.week.max()].customer_id.unique()
    else:
        customers = transactions.customer_id.unique()
    purchase_history = transactions.groupby('customer_id')['article_id'].apply(list)
    # turn articles into indices
    purchase_history = purchase_history.apply(lambda x: [article_map[a] for a in x])
    purchase_history = purchase_history.to_dict()

    result = [pd.DataFrame({'customer_id': customer,
                            'article_id': article_ids[np.argsort(sim[purchase_history[customer]].mean(axis=0))[-n:]],
                            })
              for customer in tqdm(customers, desc=f"Getting top {n} with {sim_type} similarity", leave=False)]

    # result = []
    # pool = Pool(7)
    # for r in tqdm(pool.starmap(get_similar_items_customer, [(customer, sim, article_ids, purchase_history, n)
    #                                                        for customer in customers]),
    #               desc=f"Getting top {n} with {sim_type} similarity", total=len(customers), leave=False):
    #     result.append(r)

    result_df = concat_downcast(result)
    result_df.reset_index(drop=True, inplace=True)

    return sim, article_map, result_df


# def get_similarity_samples_w(data, transactions, n, sim_type, unique_transactions, test_set_transactions, w):
#     custumers_to_use = unique_transactions if w < transactions.week.max() + 1 else test_set_transactions
#     # get more samples for latter weeks, less for earlier weeks
#     i = int((w - transactions.week.min()) / (transactions.week.max() + 1 - transactions.week.min()) * n)
#     i = max(1, i)
#     transactions_w = transactions[transactions.week < w]
#     sim_matrix, article_map, sim_sam = get_similar_items(data['articles'], transactions_w, i, sim_type,
#                                                          last_week_customers_only=True)
#     sim_sam['week'] = w
#     sim_sam = merge_downcast(custumers_to_use, sim_sam, on=['week', 'customer_id'])
#     return sim_sam

def get_similarity_samples(data, transactions, n, sim_type, unique_transactions, test_set_transactions):
    """
    Get negative samples for each week
    Get candidates for prediction
    Based on similarity of articles
    :param data:
    :param transactions:
    :param n:
    :param sim_type:
    :param unique_transactions:
    :param test_set_transactions:
    :return: similarity matrix, dict mapping article ids to matrix indices, dataframe with similar items
    """
    previous_week_info = data['article_week_info']
    samples = pd.DataFrame()
    sim_matrix, article_map = None, None
    for w in tqdm(range(transactions.week.min() + 1, transactions.week.max() + 2), desc=f"{sim_type} samples per week"):
        customers_to_use = unique_transactions if w < transactions.week.max() + 1 else test_set_transactions
        # get more samples for latter weeks, less for earlier weeks
        i = int((w - transactions.week.min()) / (transactions.week.max() + 1 - transactions.week.min()) * n)
        i = max(1, i)
        transactions_w = transactions[transactions.week < w]
        sim_matrix, article_map, sim_sam = get_similar_items(data['articles'],
                                                             transactions_w,
                                                             i,
                                                             sim_type,
                                                             last_week_customers_only=True)
        sim_sam['week'] = w
        sim_sam = merge_downcast(customers_to_use, sim_sam, on=['week', 'customer_id'])
        samples = concat_downcast([samples, sim_sam])

    samples = merge_downcast(samples,
                             previous_week_info[['week', 'article_id', 'price']],
                             on=['week', 'article_id'], how='left')
    return sim_matrix, article_map, samples
