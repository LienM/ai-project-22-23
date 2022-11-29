The code is based on https://github.com/Wp-Zhang/H-M-Fashion-RecSys (high leaderboard score, nice notebook structure, preprocessing steps).

Project Organization
------------
    ├── models             <- Trained model.
    │
    ├── notebooks          <- Jupyter notebooks, the main notebook for the project
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        ├── utils.py       <- Useful scripts for data processing 
        │
        ├── data           <- Scripts to preprocess data (there are )
        │   ├── datahelper.py 
        │   └── metrics.py
        │
        ├── features       <- Scripts of feature engineering 
        │   ├── base_features.py  (features from initial notebook)
        │   └── extra_features.py (my extra features)
        │
        └── retrieval      <- Scripts to generate candidate articles for ranking models
            ├── collector.py
            └── rules.py

--------
