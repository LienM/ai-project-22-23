# Final code

## LSTM
The directory `lstm` contains the LSTM approach as explained in the 4th lecture. This approach is no longer used after 
this lecture. However it has been cleaned up so I decided to not throw away this work.

## Final
This directory (`final`) contains all code related to the image similarity approaches. This final approach is based
on a notebook that can be found here: https://www.kaggle.com/code/marlesson/building-a-recommendation-system-using-cnn-v2/notebook

In what follows, I will describe 
all files within this directory, but I will start with an overview of the implemented pipeline and 
a short explanation of the output directory structure.

### Pipeline
The basic pipeline is:
```
preprocessing.ipynb >>> embeddings.py >>> similarities.py >>> recommendations_pairwise/mixed_recency.py >>> kaggle_submit.py
```

### Output directory structure
Output is written to a specified directory. This can be anything as long as the path is adjusted in `functions.py`.
When executing all files listed in the upcoming sections, a structure with 4 subdirectories will exist: 
- `embeddings`
- `similarities`
- `recency_pairwise_prediction`
- `mixed_prediction`

Each of them has multiple subdirectories;
- `embeddings` has all subdirectories of the form 
  - `embeddings_MODEL_W128_H128`
  - `extended_embeddings_MODEL_W128_H128`
- `similarities` has all subdirectories of the form 
  - `similarities_MODEL_W128_H128` 
  - `extended_similarities_MODEL_W128_H128`
- `recency_pairwise_prediction` has all subdirectories of the form
  - `recency_pairwise_prediction_MODEL_W128_H128_sim_SIM` 
  - `recency_pairwise_prediction_extended_MODEL_W128_H128_sim_SIM`
- `mixed_prediction` has all subdirectories of the form 
  - `mixed_prediction_MODEL_W128_H128_num_NUM_hist_12`
  - `mixed_prediction_extended_MODEL_W128_H128_num_NUM_hist_12`

Hereby:
- `MODEL` is the name of a model (see `embeddings.py` section)
- `SIM` is a similarity level (see `recommendations_pairwise_recency.py` section)
- `NUM` is the number of pairwise to take (see `recommendations_mixed_recency.py` section)
- `W128` and `H128` were added in the beginning to indicate with which image size I was working (128 x 128), but as I only
used this size further on, this is not needed anymore. However, removing it is an ideal recipe for disasters.
- `hist_12`: same as `W128` and `H128`, good intentions but not needed anymore.

In short, the name of a directory is directly related to the model that was used to create the embeddings.


### Files
#### preprocessing.ipynb
At first, it is important to note that my `preprocessing.ipynb` relies on a modified version of the preprocessing 
(`simplifying.py`) Thomas Dooms shared in Discord. The following changes were made: I didn't encode the `customer_id` as 
an integer as this was more complex for application in my code than it should be as a simplification, and I didn't drop columns when 
simplifying articles. I did not include this file in my code as it is not mine and the changes are only minor.

My preprocessing involved:
- Articles:
  - Adding a column with the image name for each article (if this image exists). This name is based on the article id.
  - Adding popularity to the article dataframe. I don't think I still use it though.
- Customers:
  - Creating a purchase history from the transactions dataframe.
  - Storing the age of the customer in this new frame (needed for popularity baseline).
- Cold start:
  - Creating an age-based popularity baseline. For each age, a window of 5 years was defined (age - 2 --> age + 2) and
  the most popular items within this age group are stored for that age.

#### batch_process.py
This file is a python file containing a `BatchProcess` class, from which multiple other classes in other files 
are derived. It provides a general framework for applying calculations to a dataframe in batches. This includes progress
printing, joining of subfiles etc. 

The idea is as follows:
1. The user defines the batch size and some other parameters (depending on the subclass).
2. The data is divided into batches of the specified size.
3. The subclass implements a batch function, this is the function to be called on each batch.
4. The batches are processed sequentially (I removed all multi-threading stuff, which makes it more readable).
5. In each batch, one or more output files are generated. These contain the output for that batch.
6. Afterwards, the output files of all batches are joined into a single output file.
7. Finally, the single output file is moved to a new directory whose name is mostly a combination of the `.py` filename 
of the subclass, the model being used to create the embeddings and width and height of the images used to create the embeddings. 
The previous section provided more detail.

The reason I chose for this class and all of its subclasses to be in a `.py` file rather than a notebook is related
to memory. Most processes took almost all my RAM while running in a notebook, which was no longer the case in a `.py` file 
(this allowed it to run large experiments in the background).
Before moving to `.py`, I tried to include `del` statements for objects that were not needed anymore, but this was not sufficient.
Notebooks are great for smaller things and experiments, but when it comes to handling entire dataframes in a large experiment 
such as my approach to the problem statement, I still see more advantage in `.py` files.

#### embeddings.py
Image similarity all starts with the creation of embeddings for each image. Therefore, the file `embeddings.py` contains
an `EmbeddingCalculator` class that inherits from `BatchProcess`. Given the name of a pretrained model, embeddings are
obtained by generating predictions from the specified model. As some articles do not have an image, their embedding is defined
as an embedding of the same shape with zeros only. 

Executing this file can be done as follows:
```
python embeddings.py --model X -- nr_rows_per_batch Y,
```
where `X` is the name of the pretrained model to use and `Y` is the number of rows per batch (default 1000).
The supported models are the top 11 models from https://keras.io/api/applications/, being:

| Model name        |Embedding shape|
|-------------------|---|
 | ResNet50          |2048|
 | ResNet50V2        |2048|
 | ResNet101         |2048|
 | ResNet101V2       |2048|
 | ResNet152         |2048|
 | ResNet152V2       |2048|
 | InceptionV3       |2048|
 | InceptionResNetV2 |1536|
 | VGG16             |512|
 | VGG19             |512|
 | Xception          |2048|


#### extended_embeddings.py
In `extended_embeddings.py`, the embeddings created in `embeddings.py` are extended with a Word2Vec embedding of the
article description. This embedding has a fixed shape, being 300. As the embeddings have either shape 2048, 1536 or 512, 
this means that the embeddings now have shape 2348, 1836 or 812 (respectively), which makes the newly added embedding 
respectively take 12.8%, 16.3% or 36.9% of the entire extended embedding.

The `extended_embeddings.py` contains an `ExtendedEmbeddingCalculator` class which inherits from `BatchProcess`. This
class first calculates the Word2Vec embeddings for the article descriptions and then loads a by the user specified 
embedding file, divides it in batches and concatenates the image embeddings with the Word2Vec embeddings (over axis 1).
As a last step this new (extended) frame is written to a new file.

Executing this file can be done as follows:
```
python extended_embeddings.py -- nr_rows_per_batch Y,
```
where `Y` is the number of rows per batch (default 1000).


#### similarities.py
If the embeddings have been created, it is time to define similarity between embeddings and thus between images. The 
`similarities.py` file has a `SimilarityCalculator` class which, again, inherits from `BatchProcess`. It reads an 
embedding file, divides it into batches (by definition of `BatchProcess`) and then passes each batch trough Sklearn's 
`cosine_similarity` function. The similarity of each article in the batch to each article in the entire frame is 
calculated. For each of the articles, only the 250 most similar articles are kept
as it is impossible to store a frame of shape 105542 x 105542, and the less similar articles may not be that relevant to 
store. Three intermediate files are written for each batch: one containing the (highest) similarities 
themselves, one containing the indices of the articles that are most similar to an article from the batch and one 
containing the article id's (being the article id on the index specified in the index output frame). 

Executing this file can be done as follows:
```
python similarities.py --embedding_version X -- nr_rows_per_batch Y,
```
where `X` is the name of the directory containing an `embeddings.feather` file and `Y` is the number of rows per batch 
(default 1000).


#### recommendations_pairwise_recency.py
This file is one of the two files that create submissions. The class `RecencyPairwiseRecommender` inherits from `BatchProcess`.
It reads a by the user specified 'similarity version' and takes a parameter `sim_level`. Then, the following mechanism
occurs: 

*For each user, take the 12 most recent purchases ('recency') and for each of these purchases ('pairwise'), take the 
article that is sim_level-th most similar.*

Thus, if `sim_level` equals 1, the most similar item is taken for each of the 12 most recent purchases, if it is 2, the
2nd most similar item is taken and so on. For users with a purchase history of less than 12 articles, the output is 
supplemented with the popularity baseline as described earlier. Thus, a user with too little purchases gets
additionally recommended a subset of the 12 most popular articles in his/her age group. Finally, the output is checked (is this a 
valid submission?) and written to file, ready to submit. 

Executing this file can be done as follows:
```
python recommendations_pairwise_recency.py --similarity_version X -- nr_rows_per_batch Y --sim-level Z,
```
where `X` is the name of the directory containing similarity files, `Y` is the number of rows per batch 
(default 1000) and `Z` is the similarity level as described in the previous paragraph.

#### recommendations_mixed_recency.py
This file is the second file creating submissions. The class `RecencyMixedRecommender` inherits from `BatchProcess`. It
is very similar to pairwise recency. At first the algorithm was rather complicated in the form of a voting system:

*For each user, take the m most recent purchases and for each of these, take the n most similar items. Add the 12 articles
from the popularity baseline (see previous section) as candidates too. Then, for each of
the m x n + 12 similar articles, get the similarity to each of the m most recent purchases (use 0.5 if an article is not 
within the 250 most similar articles of a purchased article). Return the 12 articles with the highest overall similarity.*

This seemed to work fine, my score was low but higher than my LSTM approach, so I was satisfied. This changed when I 
accidentally discovered there was not really a voting system. Due to a bug, all similarities were set to 0.5. Pandas did
a sort operation on a dataframe with values that were all the same. This was not really a 'sort', but the values swapped
such that some of the most recent items were recommended together with a part of the popularity baseline. Fixing the bug
scored less good on Kaggle, so the initial idea was a bad idea. But, by looking at the 'sort', I came on the idea of
mixed recency, which is actually nothing more than pairwise recency on the `Z` most recent articles, supplemented with
the popularity baseline. Changing my code to this idea gave similar scores. As painful as it may be, `Z=0` returned the
highest score (0.0056). In other words: the baseline is performing better than my ideas. Auwtch. 

In short, the current mixed recency idea is thus:

*For each user, take the Z most recent purchases and predict the most similar items for each of these. Add the 12 - Z
most popular items from the earlier described popularity baseline to get 12 recommendations in total.*

Executing this file can be done as follows:
```
python recommendations_mixed_recency.py --similarity_version X -- nr_rows_per_batch Y --num__similar Z,
```
where `X` is the name of the directory containing similarity files, `Y` is the number of rows per batch 
(default 1000) and `Z` is the similarity level as described in the previous paragraph.

#### kaggle_submit.py
With this file, I made nearly all my submissions.
You as a user need to give a few parameters that define which files will be submitted. These files are then, one by one,
converted to a `.zip` archive and then via the Kaggle API pushed to the origin. As my final result table contains the
numbers of 506 submissions, this file was necessary. I do not have the patience to upload the files one by one. Having
said this, it is a small wonder that Kaggle on my last submission still gave no signs of hating me. 

#### visualize.ipynb
This notebook is designed to visualize predictions. There are two options:
1. Either you choose for random visualization. In this case, a customer is picked at random, the 12 most recent purchases are
showed together with the recommendations of a preset submission file. 
2. Or you choose for visualization of a specific customer. In this case, you specify either the index of a customer in the
dataframe or the customer id, and the images as described in 1. are displayed for this user.

#### tsne_evaluation.ipynb
This notebook is intended to see how the embeddings are related to certain features in the dataframe. With TSNE, the
high-dimensional embeddings are reduced to only two dimensions, and they are plotted with respect to some categorical features
such as `product_group_name`.

#### plot.ipynb
In this notebook, the results of all submissions are visualized by means of plots. The results of the submissions can be
found in `kaggle_results` as `.csv` files. Only the public scores are plotted, although the private scores are included
in the `.csv` files as well. 

#### fuctions.py
Just a file with all kinds of helper functions.

### Submissions
As already mentioned, I did 506 submissions.
- There are 11 models, I wanted to test them all in detail.
- Regular vs extended (brings to 22)
- Pairwise recency: took the first 10 similarity levels (22 x 10 = 220)
- Mixed recency: wanted all mixes [0-12, 1-11, 2-10,...,12-0] (22 x 13 = 286)
- Thus in total: 506 submissions

### Tests/checks
The code has several asserts, mostly in the context of `BatchProcess` (in the base class and its subclasses). 
One must be sure that no data gets lost or is duplicated within the process of embedding-, similarity- or submission 
construction. 

There is also a "check submission" assert in the both submission creation files. The associated function checks whether
there is a submission for each customer that has exactly 12 recommendations. Furthermore, it checks the number of columns
in the submission file as it is so easy to forget `index=False`. Unfortunately, I didn't check whether the 12 recommendations
for a user are all unique... Found this too late so there was no time to redo everything. 

