import filecmp

import numpy as np
import pandas as pd


def check_file_similarity(fname1, fname2):
    return filecmp.cmp(fname1, fname2)


if __name__ == '__main__':
    fname1 = 'out/age_bin_prod_week_106_bin_size_1.csv.gz'
    fname2 = 'out/age_bin_prod_week_106_bin_size_32.csv.gz'
    print(check_file_similarity(fname1, fname2))
    print(pd.read_csv(fname1).head(20))
    print(pd.read_csv(fname2).head(20))


def cosine_similarity(vec1, vec2):
    """
    calculate the cosine similarity between two vectors
    :param vec1: first vector
    :param vec2: second vector
    :return: cosine similarity between vec1 and vec2
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def make_low_high_interval(interval):
    """
    make an interval only containing max and min from given interval list
    :param interval: interval to make low high interval from
    :return: the low high interval
    """
    return [min(interval), max(interval)]
