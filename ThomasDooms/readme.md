# Thomas Dooms
S0181389

---

# Disclaimer
I know that everyone is using notebooks, but I don't like them, I prefer structuring my code myself.
Additionally, notebooks are not properly supported by PyCharm which leads to it feeling very clunky.
And lastly, I need to work on my laptop quite frequently where it takes a lot of time to process stuff,
so I save intermediate results to disk and share them on a private cloud to use on my laptop.
Because I have to do this manually anyway, there is no reason for me to use notebooks.

# Structure

I hate large files, I hate big folders. This is why I have lots of files. 
In the next section I will explain the general structure of my project and what file contains what.

### Data
Contains the raw data from Kaggle as well as intermediate/processed data files.
Data is structured per dataset or category, it contains the following subfolders:
- `articles/customers/transactions`: Contains original, simplified, feature engineered and subset .feather files.
- `candidates`: Kind obvious, contains several generated candidates
- `models`: Contains the trained models which can be used with different baseline for example
- `submissions`: Contains the example submission for the Kaggle competition and a simplified version of it

This folder is not pushed to GitHub as it contains too much data.
This is jsut to give an insight without running the code yourself.

### Experiments
Sometimes I just need a place to put some code to test something out. 
These files are not as well documented than my main code as most of the time it's simple data analysis.
These files are also not structured very well as it's more of a playground. 
I kind of use these as a notebook where I comment out code and try out different things.

This contains my first experiments with:
| File                | Description                                                                           | Lecture |
| ------------------- | ------------------------------------------------------------------------------------- | ------- |
| clustering          | File to cluster articles and do analysis on the clusters based on seasonality         | 5       |
| datetime old        | Simple pandas stuff with date & time (for learning purposes)                          | 3       |
| lecture4 old        | Old solution where I decided to start from scratch                                    | 4       |
| postal              | Simple file to get statistics about the postal codes                                  | 5       |
| pytorchgeom         | Contains the code for training node2vec models with pytorch geometric                 | 4       |
| rolling in the deep | Initial experiments with networkx and karate club using deepwalk                      | 4       |
| sampling            | Old code which made sampled datasets for quick testing                                | 2       |
| sbert               | Sentence transformers (sbert) to embed text                                           | 3       |
| seasons             | Simple plotting and data analysis of the seasonality of the data                      | 5       |
| so annoying         | Experiments with annoy to find nearest neighbors in the graph embeddings              | 4       |
| swin                | Swin transformers to embed images (I quickly abandoned it as I found another topic)   | 3       |
| test                | Old code to test some stuff, this was frequently changed to test small things         | 3       |

The most interesting files are probably `clustering` and `pytorchgeom` as I put the most work into them.
The first contains a very detailed analysis of the seasonality of the data and the clustering of the articles which I found fascinating.
It was one of the times when I really felt like I was learning something new and making good decisions and getting perfect results.

The second file contains the code for training node2vec models with pytorch geometric. 
The code itself is not spectacular but this took about 2 full days of research to pick the right methods, understand them and find a fitting library.
As described above I first used karate club as it seemed like a good starting point, but I quickly found out that it was not the right tool for the job.
It didn't use any multithreading or gpu acceleration meaning it would take ~30 hours on a graph with 1.4 million nodes to train a model.
Using pytorch geometric was actually quite difficult as it is mostly used for graph neural networks (classification) and not for node2vec.


### Previous
These are the first notebooks from the really early 
lectures (eda/classification) which are not really used anymore.

These were from dark times when I was still learning how to use pandas, numpy and doing basic recommender systems.
This is why I won't really cover it here.

### Src
This is the meat of the project, it contains most of the work I did for the lectures.
It is structured as follows:
- `candidates` Functions to generate candidates with multiple methods
- `features` Functions to generate features for each dataset
- `infer` Single function to infer the submissions given a baseline, data and other stuff
- `info` Again, similar to a test file, I used it to gather very simple statistics or to remind me what columns there are
- `main` The main file, this is where the magic happens, more on this below
- `paths` Contains a single function with a helper function for my data file paths
- `simplify` This is a file I shared with the clas, it contains rudimentary preprocessing and strong reduction of data size
- `train` Has a simple function which trains a LightGMB model and saves it to disk

My idea was to have a main file which would be a single-press pipeline to run the whole project.
It would check which files exist and which still need to be generated. 
This, actually, was way more involved than I originally thought to implement efficiently.
One reason is that I wanted to avoid one stage (features) to write a file and that the next stage (candidates) would read it again.
This in combination with a seamless integration of test runs that I wanted, made it complicated.
After a few hours, I decided to just leave it be and rerun the stages I needed manually.

# Code sharing
I shared my simplify code with the class as a shared baseline for features.
This had the advantage of correctly implemented size reduction which made it possible for everyone to run it locally.
I feel like this should be encouraged more in next iterations, where certain functions can easily be used by everyone.

# Code giveaway
Between lecture 4 and 5 I noticed that I still had lots of inspiration for new things I could try. 
When Basil Rommens approached me with the question of what he could do, 
I suggested he take my code to finish it as I wasn't sure I was going to have the time.
I really liked the idea of graph embeddings and learned a lot from it, so I was happy to share my code with him.