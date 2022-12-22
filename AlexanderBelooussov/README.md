# AI Project: Alexander Belooussov
___
## Main Python Files
As you can see, I chose to not use notebooks for this project. Instead, I decided to split my code into several python files. Code can be found in the src directory. Below I list a brief description of each file's function.

### Main
File that coordinates execution of the recommender, as well as cross validation tests. Execution can be controlled using command line parameters

* `--n_train_weeks`: Amount of weeks to use to train the model. Statistical data will still be generated on all weeks, and algorithms using purchase history such as ItemKNN will use 2 * n_train_weeks to ensure that enough context is provided to the earlier weeks. default=12
* `--n`: Amount of samples and candidates to generate for each method. For example: customer X buys in week Y and this is a positive sample. N negative samples will be generated for week Y and customer X for each method. Additionally, N candidates get generated for customer X in the test week for each of these methods. default=12
* `--frac <float>`: The fraction of customers to use. This is useful for quick tests on a sample dataset. default=1.0
* `--ratio <int>`: The amount of negative samples to use for every positive sample. This is an upper limit, and may not be reached if not enough negative samples are generated. default=1
* `--n_estimators <int>`: Amount of trees to use with LightGBM. default=100
* `--cv`: Whether to leave out the last week and use it for validation. Allows calculating the MAP@12 and recall for the last week.
* `--verbose`: Whether to print more output
* `--itemknn`: Whether to generate samples and candidates using ItemKNN
* `--l0`: Whether to generate samples and candidates using l0-norm or hamming distance to previously purchased items to generate samples and candidates
* `--w2v`: Whether to generate negative samples and candidates using Word2Vec similarity to previously purchased items to generate samples and candidates
* `--p2v`: Whether to generate negative samples and candidates using Prod2Vec
* `--random`: Whether to randomly generate negative samples
* `--grid`: Perform a random grid search. This makes a grid of some possible parameters and runs cross validation on a random subsection. All other parameters except --frac are ignored.

### Preprocessing
This file takes care of preprocessing and writing feather files.
Feather files are created of unprocessed data to better facilitate any changes in preprocessing.

### Rank
This file takes care of training the model and ranking the candidates.

### Random samples
This file generates the random samples

### Recpack samples
This file generates the samples and candidates using ItemKNN or Prod2Vec, implemented in RecPack. It also adds the associated scores from the algorithms to the final set of samples and candidates

### Similarity
This file generates the samples and candidates using l0-norm/hamming distance and Word2Vec similarity. It also adds the associated scores from the algorithms to the final set of samples and candidates

### Samples

Main file for generating negative and positive samples, as well as candidates.

### Utils
A file containing utility functions such as loading data, writing feather files, merging/concatenating dataframes, etc.

### Historical info
This file generates statistics for articles and customers based on the given transactions.

___
## Exploratory Data Analysis
Unfortunately I deleted the file before it was said that it should also be submitted. :(

## Lecture 2 Notebook
The code used for the second lecture can be found in lecture2.ipynb. This code was not used for later parts of the project and is thus left as is.