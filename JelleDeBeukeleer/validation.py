import pandas as pd
import json
import matplotlib.pyplot as plt

from ensemble_model import make_model, CustomEnsemble


def plot_scores(weeks: list, scores: list, filename: str = None):
    plt.xlabel("amount of preceding training weeks")
    plt.ylabel("validation score")
    plt.title("validation scores by preceding training weeks")
    plt.plot(weeks, scores)
    plt.xticks(weeks)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
    plt.cla()


def plot_features(weeks: list, feature_maps: list, filename: str = None):
    plt.xlabel("amount of preceding training weeks")
    plt.ylabel("importance per feature")
    plt.title("feature importance by preceding training weeks")
    x = weeks
    y_list = [[] for i in feature_maps[0].keys()]
    y_labels = list(feature_maps[0].keys())
    for f_map in feature_maps:
        for i, key in enumerate(f_map.keys()):
            y_list[i].append(f_map[key])
    for i, y in enumerate(y_list):
        plt.plot(x, y, label=str(y_labels[i]))
    plt.xticks(weeks)
    plt.legend(loc="best")
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
    plt.cla()


def validate_sliding_window(df: pd.DataFrame, training_weeks: int = 1):
    transaction_weeks = df.t_dat.unique()
    prediction_week = transaction_weeks.max()
    scores = []
    min_week = transaction_weeks.min()
    model = None
    for week in range(min_week, prediction_week - training_weeks+1):
        model = make_model(df)
        training_window = [week + i for i in range(training_weeks)]
        validation_week = training_window[-1] + 1
        training_transactions = df[df.t_dat.isin(training_window)].reset_index()
        validation_transactions = df[df.t_dat == validation_week].reset_index()
        if "index" in training_transactions.columns:
            training_transactions.drop("index", inplace=True, axis=1)
            validation_transactions.drop("index", inplace=True, axis=1)
        model = model.fit(training_transactions.drop("ordered", axis=1),
                          training_transactions["ordered"])
        score = model.score(validation_transactions.drop('ordered', axis=1),
                            validation_transactions["ordered"])
        scores.append(score)

    if len(scores) == 0:
        # can not properly validate on empty set
        return 0
    if model:
        print("feature weights:", model.feature_importances_())
    mean_score = sum(scores) / len(scores)
    return mean_score, model.feature_importances_()


if __name__ == "__main__":
    settings_file = "./settings.json"
    settings = json.load(open(settings_file))
    data_dir = settings["data_directory"]
    processed_fn = settings["data_filenames"]["processed"]
    recent_weeks = settings["recent_weeks"]
    transactions = pd.read_csv(data_dir + processed_fn["transactions"])
    full_weeks = settings["full_weeks"]
    view = (transactions.t_dat.max() - transactions.t_dat) < full_weeks
    transactions = transactions[view]
    transactions.drop(["customer_id"], inplace=True, axis=1)
    weeks_range = range(1, len(transactions.t_dat.unique()))
    scores = []
    FA = []
    for weeks in weeks_range:
        score, feature_importances = validate_sliding_window(transactions, training_weeks=weeks)
        print(weeks, score)
        scores.append(score)
        FA.append(feature_importances)

    plot_scores(weeks_range, scores, "scores.png")
    plot_features(weeks_range, FA, "features.png")
