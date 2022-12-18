from functions import *


"""
Do Kaggle API submissions in an organized fashion, for a series of csv files.
"""


if __name__ == '__main__':
    # comment the models you do not want to submit
    models = [
        'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'InceptionV3',
        'InceptionResNetV2', 'VGG16', 'VGG19', 'Xception'
    ]

    # base directory: either recency_pairwise_prediction or mixed_prediction
    dir_name = 'recency_pairwise_prediction'

    # comment one of both if you do not want the regular/extended
    extensions = ['', 'extended_']

    # suffixes: can be _sim_{i}, but also _num_{i}_hist_12
    suffixes = [f'_sim_{i}' for i in range(1, 11)]

    progress = 1
    for suffix in suffixes:
        for x in extensions:
            for model in models:
                print(f'{progress}/{len(models) * len(extensions) * len(suffixes)}', model, x, suffix)
                progress += 1

                # define the csv filename
                filename = odp(f'{dir_name}/{dir_name}_{x}{model}_W128_H128{suffix}/{dir_name}.csv')

                try:
                    # compress the csv file to a zip archive
                    os.system(f'zip -j {filename.replace(".csv", ".zip")} {filename}')
                except Exception as e:
                    pass

                os.system(
                    # do submission
                    # --> the submission message equals the name of the directory where the csv was stored
                    f'kaggle competitions submit -c h-and-m-personalized-fashion-recommendations '
                    f'-f {odp(dir_name)}/{dir_name}_{x}{model}_W128_H128{suffix}/{dir_name}.zip '
                    f'-m "{dir_name}_{x}{model}_W128_H128{suffix}"'
                )
                print()

                # remove the cs file (for completeness)
                os.remove(filename.replace(".csv", ".zip"))
