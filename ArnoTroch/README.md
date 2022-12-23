# Final code review

All the code that has to be considered for review is in the `final` directory.

## File structure

I numbered my notebooks in the order that they were made across the lectures, which is therefore also the order that
they should be looked at. Each of these numbered notebooks has a short summary at the start to explain what exactly
happens through the course of the notebook.

There are also some normal python files. The `candidates.py` file contains functions to generate different sorts of
candidates and is used in multiple notebooks. The `helpers` directory contains the following files:

- `create_samples.py`: script that I ran to generate 5% samples of the dataset.
- `create_validation.py`: script that I ran to generate a validation set from the last week of data.
- `evaluation.py`: contains functions to calculate Average Precision@K and to check a submission file against the
  validation set.
- `utils.py`: contains functions to convert customer and article ID's.