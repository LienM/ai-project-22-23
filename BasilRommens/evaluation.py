import pandas as pd


def precision(rel_u_items, user, k):
    u_recs = user['prediction'].split(" ")
    top_k = set(u_recs[:k])
    return len(top_k.intersection(set(rel_u_items))) / k


def relevance(rel_u_items, user, k):
    u_recs = user['prediction'].split(" ")
    if len(u_recs):
        return 0
    return u_recs[k] in rel_u_items


def map_at_k(rel_items, users, n, k):
    sum_U = 0
    for user in users.iterrows():
        sum_k = 0
        uid = user[1]['customer_id']
        rel_items = rel_items[rel_items['customer_id'] == uid]
        for l in range(1, min(len(rel_items), k) + 1):
            sum_k += precision(rel_items, user[1], l) + relevance(rel_items,
                                                                  user[1],
                                                                  l)

        sum_U += sum_k
    return sum_U / (len(users))


if __name__ == "__main__":
    rec = pd.read_csv("random_rec.csv")
    rel = pd.read_csv("transactions.csv")[['customer_id', 'article_id']]
    print(map_at_k(rel, rec, 12, 12))
