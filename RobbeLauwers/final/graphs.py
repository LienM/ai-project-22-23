import matplotlib.pyplot
import seaborn as sns
import csv

def journey(full=False):
    numbers = []
    with open("score_history_full.csv") as f:
        read = csv.reader(f)
        for row in read:
            numbers.append(float(row[0]))

    print(numbers)
    sns.set_theme()
    # sns.lineplot(numbers)
    #
    # matplotlib.pyplot.show()

    if not full:
        no_outliers = [item for item in numbers if item > 0.0206]
        temp = sns.lineplot(no_outliers)
    else:
        temp = sns.lineplot(numbers)

    temp.set_ylabel("Public score", fontsize=16)
    temp.set_xlabel("Attempt", fontsize=16)
    if not full:

        temp.text(-0.5, no_outliers[0]/(min(no_outliers)+max(no_outliers)), 'Baseline', rotation=90, transform=temp.get_xaxis_text1_transform(0)[0])
        temp.text(5.5, 0.6, 'Importance', rotation=90,
                  transform=temp.get_xaxis_text1_transform(0)[0])
        temp.text(11.5, 0.4, 'Features', rotation=90,
                  transform=temp.get_xaxis_text1_transform(0)[0])
        temp.text(14.5, 0.4, 'Full dataset features', rotation=90,
                  transform=temp.get_xaxis_text1_transform(0)[0])
    matplotlib.pyplot.tight_layout()
    if not full:
        labels = [0.0206,0.0207,0.0208,0.0209,0.0210,0.0211,0.0212,0.0213,0.0214,0.0215,0.0216]
        temp.set_yticks(labels)
        temp.set_yticklabels(labels)
    matplotlib.pyplot.show()

def weeks():
    numbers = []
    weeks = []
    with open("score_weeks.csv") as f:
        read = csv.reader(f)
        for row in read:
            weeks.append(int(row[0]))
            numbers.append(float(row[1]))

    print(numbers)
    sns.set_theme()
    # sns.lineplot(numbers)
    #
    # matplotlib.pyplot.show()

    temp = sns.lineplot(x=weeks, y=numbers)
    # https://stackoverflow.com/questions/43639096/setting-the-interval-of-x-axis-for-seaborn-plot
    import matplotlib.ticker as ticker
    temp.xaxis.set_major_locator(ticker.MultipleLocator(1))

    temp.set_ylabel("Public score", fontsize=16)
    temp.set_xlabel("Weeks used", fontsize=16)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def candidates():
    import pandas as pd
    data = pd.read_csv('candidates.csv')

    sns.set_theme()
    # sns.lineplot(numbers)
    #
    # matplotlib.pyplot.show()

    temp = sns.barplot(x="features",y="public score",hue="candidates",data=data)
    sns.move_legend(temp, "lower right")
    temp.set_xticklabels(temp.get_xticklabels(), rotation=60, ha="right")
    min_value = data["public score"].min()
    max_value = data["public score"].max()
    # https://www.python-graph-gallery.com/44-control-axis-limits-of-plot-seaborn
    matplotlib.pyplot.ylim(min_value-0.00001, max_value+0.00001)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

# weeks()
# journey()
journey(full=True)
# candidates()