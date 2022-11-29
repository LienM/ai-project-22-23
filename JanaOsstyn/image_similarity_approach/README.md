# Image approach

## File overview
- `functions.py` contains a set of helper functions, accessible from all notebooks
- `preprocessing.ipynb` lists some very basic preprocessing
- `embeddings.ipynb` creates embeddings for all images
- `similarities.ipynb` calculates the similarities between any pair of articles based on their embeddings
- `recommend_recency_pairwise.ipynb` recommends articles based on the 12 most recent purchases of a user
- `recommend_voting_system.ipynb` recommends articles based on the entire purchase history of each user
- `original_kaggle_notebook.ipynb` is just the original notebook found at https://www.kaggle.com/code/marlesson/building-a-recommendation-system-using-cnn-v2/notebook 
  - All of my code is based on the pipeline in this notebook


## Pipeline
Every notebook creates output files that serve as input files to other notebooks. E.g. once the similarities have been 
calculated, you can use them in both recommend notebooks without having to run them again.

`preprocessing.ipynb` > `embeddings.ipynb` > `similarities.ipynb` > `recommend_recency_pairwise.ipynb`

`preprocessing.ipynb` > `embeddings.ipynb` > `similarities.ipynb` > `recommend_voting_system.ipynb`
