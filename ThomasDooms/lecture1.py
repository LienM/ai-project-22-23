from itertools import accumulate


def map_k(rec, truth):
    correct = list(accumulate([int(r == t) for r, t in zip(recommendations, truth)]))
    return sum((r == t) * c / (i + 1) for i, ((r, t), c) in enumerate(zip(zip(rec, truth), correct))) / correct[-1]


recommendations = [1] * 10
truth = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]

print(map_k(recommendations, truth))
