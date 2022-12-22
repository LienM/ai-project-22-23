# Project AI

Investigation of a kaggle competition on predicting the clothes a customer is
going to buy

## Table of contents

* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Dataset cleaning](#dataset-cleaning)
* [Research questions](#research-questions)
    * [Correlation](#correlation)
    * [Graph embeddings](#graph-embeddings)
    * [Age bin popularity](#age-bin-popularity)
* [Exploration](#exploration)

## General info

For this project the main goal was to try to improve the kaggle score of the
competition. This was done through the use of research questions. We came up
with those during the course of the semester. I chose to work with python files
for this project as it was the clearest way to execute this project, without
relying on notebooks. There are still 2 notebooks, but they are from earlier
lectures and can thus be ignored. The main step before answering the research
questions, was to clean the dataset. This was done in order to make it easier to
handle when you don't have a lot of memory. In order to perform our submissions
we used the kaggle command line tool to perform this. All the code to submit the
results can be found in the `submit.sh` file.

## Technologies

This project requires some technologies in order to be runnable. We used python
3.10.6. Aside from using the python libraries we also used third-party ones.
Below we list an exhaustive list of the downloaded and used libraries, and their
use. We also have a requirements.txt file with all of them listed.

* `pandas`: data manipulation, analysis and cleaning
* `numpy`: storing arrays and manipulating them
* `matplotlib`: plotting graphs
* `seaborn`: plotting graphs
* `joblib`: saving and loading models
* `tqdm`: showing progress bars
* `networkx`: creating embedding graphs
* `karateclub`: deepwalk algorithm
* `annoy`: knn indexing for graph embeddings
* `random`: random number generation
* `scipy`: ranking data in order to make a denser representation and correlation
  distance
* `sklearn`: used for label encoder and train test split
* `lightgbm`: ranking our candidates

## Setup

We assume that we are in the root of the project, unless mentioned otherwise.
Another assumption is that you require a minimum of 32GB of RAM to run this.

In order to be able to run the code you first must run this line for downloading
all the libaries:

```bash
pip3 install -r requirements.txt
```

After you've installed all the libraries it is time to make the appropriate
folders:

```bash
mkdir data
mkdir out
```

to finish the code setup you can run, to clean up the data and create samples:

```bash
python3 dataset.py
```

to run the embeddings research question results:

```bash
python3 BasilRommens/graph_embedding/embed.py
python3 BasilRommens/graph_embedding/candidates.py
```

TODO
to run the age bin popularity research question results:

```bash
```

## Dataset

The main takeaway from this datasets challenge was that the dataset was too big.
Therefore, we needed to clean it to make room for more data. The code to clean
up the dataset can be found in the `dataset.py` file. We also make samples to
make the dataset much smaller. The code for reading this dataset can also be
found in there.

The function used to clean the dataset is: `prepare_feather_datasets`

The function used to make samples is: `create_samples`

We can also load in parts or the entire dataset through the functions:
`read_data_set` and `part_data_set`.

Before you start the project, you mustn't forget to prepare the dataset. This
preparation also creates featherfiles as those are lightweight and fast to load.
Then you can create the samples. And finally you can load in part or the entire
dataset.

## Research questions

Throughout the course we needed to come up with research questions. We will
further elaborate in the following subsections. I used following research
questions:

* Is there any correlation between the cross validation and the leaderboard?
* Can we get score improvements by using graph embeddings?
    * If so, can we improve the performance through putting time inside the
      graph?
* Do we improve the score by adding most popular items within age bins as
  candidates?
    * If so, then what are optimal age bins to work with?

### Correlation

To check if there is any correlation between the cross validation and the
leaderboard, we need to use some extra annotations in the code to show where
we've taken our samples from. So, every function that
says `(correlation analysis)` in its docstring participates in the correlation
analysis. These files can also be found by looking at the first number. If it
is either 102, 103, 104, or 105 then those files can be used in the correlation
analysis. The evaluation is done in the `evaluation.py` file.

### Graph embeddings

You create embeddings of a graph using a skip-n-gram model on the random walks
of the graph. We used 3 different algorithms to create graphs. They are further
specified in the file `embed.py`. We also used set up 5 different experiments to
generate suitable candidates for recommending items for customers. They are
listed at the bottom. As we use embeddings, we also want tot find the k nearest
as fast as possible and therefore we use indexing to do this, through the use
of annoy. The embeddings are generated in `embed.py` and the indexing is done in
`index.py`. The code to generate the candidates is in `candidates.py`.

### Age bin popularity

## Exploration