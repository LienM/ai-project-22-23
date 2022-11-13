import pandas as pd

sizes = {
    'tiny': 0.01 / 100,
    'small': 0.1 / 100,
    'medium': 1 / 100,
    'large': 5 / 100,
}

datasets = ["articles", "customers", "transactions"]


def create_samples():
    for dataset in datasets:
        df = pd.read_feather(f'../data/{dataset}.feather')
        for name, size in sizes.items():
            print(f"Creating sample {dataset}_{name}")

            sampled = df.sample(frac=size).reset_index(drop=True)
            sampled.to_feather(f'../data/{dataset}_{name}.feather')


if __name__ == '__main__':
    create_samples()