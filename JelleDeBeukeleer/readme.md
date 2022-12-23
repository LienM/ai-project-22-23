# code explanation
Jelle De Beukeleer

## files to run
The files to be run are all in the current top directory,
subdirectories only contain past experiments.

most of the project follows a pipeline according to the sequence _dataset structuring -> candidate + sample generation -> predicting from samples_

Note that most filenames and parameters can be altered in _settings.json_

### data preparation
_data_preparation.py_ is the first file to be run,
it removes unused features from the transactions, as well as the oldest weeks.
Some features such as the customer id and transaction date are also transformed to allow
for easier handling later.

### Candidate and sample generation
For this part, the most relevant files are _popular_candidates.py_, _popular_candidates_age.py_ and _itemset_generation.py_.
At least one of these scripts should be run, but running both popularity-based files likely results in redundant candidates. The itemset-based candidates are generally not the same, but require a significant time to run (in the space of hours), so running this file might be time consuming. I left all files in however as I could not formally determine which popularity metric performed better.

For negative sampling there is also _expand_negative_samples.py_, which randomly inserts more negative samples counting up to about
100% of all positive examples. This file can however be re-run arbitrarily often in order to increase the fraction of negative samples.

### Predictions
There are two versions of the predictions script: _final_predictions.py_ and _final_predictions_single_run.py_. These both generally do the same, except that the single_run file keeps the intermediate predictions dataframe in memory, while the other script will repeatedly write temporary files to disk in order to save memory. If memory is sufficiently large it is likely faster to use the single_run version.
The predictions are performed on batches of 300,000 customers at once, and written to 5 temporary files which can later 
be merged using _merge_predictions.py_.


### Ensembling
The ensemble model has a fairly basic implementation, located in _ensemble_model.py_. A CustomEnsemble class holds a list of submodels, as well as the final meta model.
These submodels can be trained on a sample of the dataset if necessary, and all internal classification models can be specified manually. A make_model() function is also presented
to generate a default ensemble model. The internal models specified in this function have no real logic behind them, as I just wanted to test whether multiple sklearn models worked within the structure. 

For predictions, the model first generates scores from each submodel, and then predicts its final score based on these scores, along with the original features if specified.