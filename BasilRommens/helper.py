import filecmp

import pandas as pd


def check_file_similarity(fname1, fname2):
    return filecmp.cmp(fname1, fname2)


if __name__ == '__main__':
    fname1 = '../out/age_bin_prod_week_106_bin_size_1.csv.gz'
    fname2 = '../out/age_bin_prod_week_106_bin_size_32.csv.gz'
    print(check_file_similarity(fname1, fname2))
    print(pd.read_csv(fname1).head(20))
    print(pd.read_csv(fname2).head(20))
