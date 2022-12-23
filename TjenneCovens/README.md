# 👋 Welcome to my readme! 👋

## Where should you look at?
- My final notebook "Final.ipynb" contains most of the functionality used. This file contains the main pipeline of the recommender. It does call onto some other functions in the utils.py file, but these should be clear enough. 
- Initially everything was in regular pythons files, I left them in the '/not_final' subdir. You can look at these for reference, but they are not 'as' clean and 'up-to-date' as the notebook.

## What about the data?
- Data should be put in the '/data' subdir
- Right now it loads in the 1% sample, this can be edited in the notebook though
- Make sure that the 'materials.txt' is present in the root folder since it is required for preprocessing
- Materials source: https://en.wikipedia.org/wiki/List_of_fabrics

## What output?
- The output appears in the '/output' subdir
- The format is the '.csv.gz' as described on the Kaggle page

## Can you tell me more about this Kaggle thingy?
- No, but here is a link: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations