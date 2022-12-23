The code is based on https://github.com/Wp-Zhang/H-M-Fashion-RecSys (high leaderboard score, nice notebook structure and preprocessing steps). This original project was very useful for understanding the procedure of recommendation system training. 


# Code Organization
------------
    ├── models             <- Trained models from the first iteration.
    │
    ├── notebooks          <- Jupyter notebooks, all notebooks for the project
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        ├── utils.py       <- Useful scripts for data processing 
        │
        ├── data           <- Scripts to preprocess data
        │   ├── datahelper.py 
        │   └── metrics.py 
        │
        ├── features       <- Scripts for feature engineering 
        │   ├── base_features.py  (features from initial notebook)
        │   └── extra_features.py (my extra features)
        │
        └── retrieval      <- Scripts to generate candidate articles for ranking models
            ├── collector.py
            └── rules.py

--------

# Project orgalization

Overall, there were three main steps during the project: 
1. First analyses and trials: [EDA](notebooks/EDA.ipynb), [Feature Engineering](notebooks/FeatureEngineering.ipynb), [first trials](notebooks/lecture4.ipynb)
2. Sampling analyses: Random samples and positive rate analyses - [Iteration1](notebooks/iteration1.ipynb)
3. Sampling analyses: positive samples, candidates from only one candidate generation strategy, stacking models, training dataset size analyses - [Iteration2](notebooks/iteration2.ipynb)

## Comments to notebooks 
- For the first iteration ([Iteration 1](notebooks/iteration1.ipynb)) I had a lot of issues with memory usage so I did not optimize the process of datasets creation because I ran it by parts several times. 
- The structure of the notebook for the first iteration:
  - Generate candidates 
  - Select candidates for training
  - Calculate features for selected candidates 
  - Train models and save them 
  - Validate results on the last week of available data (without cross-validation) and on Kaggle data 
- For the second iteration ([Iteration 2](notebooks/iteration2.ipynb)) I tried to ivercome my previous issues so I used other candidate generation ideas, other features, different order of calculations. 
- The structure of the notebook for the second iteration:
  - Generate candidates 
  - Calculate features for generated candidates
  - Select candidates for training 
  - Train a model 
  - Validate results on the last week of available data (with cross-validation on three weeks) without validation on Kaggle data (there were too many models for validation (including cross-validation)) 
- A set of features for two iterations is a bit different.
- Most of helping functions were used without changes in code from orginal notebook. However, sometimes I changes parameters inside functions and the procedure itself. Overall, I went through every function from original notebook that I used in my procedure and was able to update them for my needs. 
- I did not upload trained models from the second iteration because there are too many files.   
- [first trials](notebooks/lecture4.ipynb) notebook does not have a lot of comments but most of ideas from that notebook were used later in two iterations so this notebook is not very interesting at the end.  
