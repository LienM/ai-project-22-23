import sys
from batch_process import BatchProcess
from functions import *


class RecencyMixedRecommender(BatchProcess):
    """
    Subclass of BatchProcess.
    Predict articles for each customer.
    The 12 most recent purchases are extracted for each customer. Then, for the Y most recent purchased articles, the
    most similar article is collected. The 12 - Y remaining spots for recommendations are filled with the most popular
    items among customers within the same age category (see preprocessing.ipynb for a definition of this).

    Note: the purpose of this class was originally more complex. Instead of the 12 most recent articles, the X most
    recent articles were fetched, after which for each of these the Y most similar articles were collected. The 12 most
    popular articles among people with approximately the same age were added as candidates too. Then, for each of the
    candidates, the similarities to each of the X purchased articles were fetched from a dataframe with similarities.
    However, this dataframe only stores the 250 most similar items, which means it was possible for a candidate to have
    no similarity value with respect to a certain purchase. In this case, 0.5 was stored. From this point, the goal was
    to sort the similarities and return the 12 articles that were, in total, most similar to all X purchased items.
    This seemed to work very well (> 0.004, +-double of just pairwise recency) but accidentally it came out that the
    similarity shortcut dictionary that was built hadn't the right keys, meaning all similarities were 0.5 at sorting
    time. Pandas performed Quicksort and thus modified the array slightly. From this it came out it gave me the > 0.004
    scores just because he took the 2 recommendations for the last 2 purchases and 10 of the most popular baseline.
    After fixing this, the score dropped to 0.002, which made me throw away the fancy sorting idea as it performed worse
    than the much simpler idea of mixing recency with popularity.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        """
        Constructor
        :param args: a list of arguments that will be processed by self.read_arguments.
        """
        BatchProcess.__init__(self, modulename='mixed_prediction')

        # SIMILARITIES
        self.num_similar = None                         # how many similar values to take for each purchased article
        self.similarity_version = None                  # similarity version (relates to embedding version)
        self.similarity_value_df = None                 # a dataframe holding the similarity values
        self.similarity_ids_df = None                   # a dataframe holding the corresponding article ids

        # TRANSACTIONS
        cold_start_df = pd.read_feather(odp(filename='cold_start_recommendations.feather'))
        cold_start_df['most_popular'] = cold_start_df.drop(columns=['age']).astype(str).agg(' '.join, axis=1)
        cold_start_df = cold_start_df[['age', 'most_popular']].copy()
        # a dataframe holding all customer transactions as well as the relevant cold start for each customer (age-based)
        self.customers_transactions_df = pd.read_feather(idp(filename='customer_transactions_processed.feather'))
        self.customers_transactions_df = self.customers_transactions_df.merge(cold_start_df, how='left', on='age')
        # nr rows = nr of customers to find predictions for
        self.nr_rows = self.customers_transactions_df.shape[0]

        # ARTICLES
        article_df = pd.read_feather(idp(filename='articles_processed.feather'))
        # a list of article_ids for which no image exists
        self.no_img_articles = article_df[article_df['image_name'] == 'does not exist']['article_id'].values.tolist()

        # other
        self.base_filenames = ['mixed_prediction']      # names for final and temporary files

        self.read_arguments(args)                       # read the arguments

    # ------------------------------------------------------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------------------------------------------------------

    def print_help(self):
        """
        Prints some information for the user about the parameters that should be provided.
        This method is called if either:
            - the user passed 'help' as one of the arguments
            - the user provided a keyword that couldn't be recognized by the subclass
        """
        print('python recommendations_mixed_recency.py followed by:')
        print('\t--similarity_version value (required)')
        print('\t--num_similar value (optional, default 3)')
        print('\t--nr_rows_per_batch value (optional, default 10000)')
        exit()

    def read_arguments(self, args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        The arguments should include:
            - '--similarity_version' followed by the name of a directory containing similarity feather files (required)
            - '--num_similar' followed by an int, indicates how many candidates per purchase
            - '--nr_rows_per_batch' followed by the batch size (number of rows to be considered per batch)
        If 'help' is included in args or an unknown argument is found, self.print_help is called.
        The value of self.output_directory is based on the model name, and the image width and height.
        :param args: a list of arguments
        """
        if 'help' in args:
            self.print_help()

        for i, arg in enumerate(args):
            if arg == '--num_similar':
                self.num_similar = int(arguments[i + 1])
            elif arg == '--nr_rows_per_batch':
                self.nr_rows_per_batch = int(arguments[i + 1])
            elif arg == '--similarity_version':
                self.similarity_version = arguments[i + 1]
            elif not arg.startswith('--'):
                continue
            else:
                print(f'Unknown argument {arg}, expected:')
                self.print_help()

        # now that the similarity version is known, the similarity dataframes can be loaded
        self.similarity_ids_df = pd.read_feather(
            odp(f'similarities/{self.similarity_version}/similarities_ids.feather')
        )
        self.similarity_ids_df = self.similarity_ids_df.set_index(self.similarity_ids_df['article_id'])
        self.similarity_value_df = pd.read_feather(
            odp(f'similarities/{self.similarity_version}/similarities_values.feather')
        )
        self.similarity_value_df = self.similarity_value_df.set_index(self.similarity_value_df['article_id'])

        # also, the output directory can now be defined
        self.output_directory = f"{self.modulename}_{self.similarity_version.replace('similarities_', '')}_" \
                                f"num_{self.num_similar}"

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method forms an extension on the self.can_run of the superclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.num_similar is not None and \
            self.similarity_version is not None and \
            self.similarity_value_df is not None and \
            self.similarity_ids_df is not None

    def run(self):
        """
        The run() method of the base class is executed as before, but now we need to postprocess the feather output.
        Kaggle expects csv (or zip,...), not feather so the feather file should be read in and written out to csv.
        As this class is expected to output a submission file, it is a good idea to check whether the submission file is
        valid (contains right # columns and rows, header is correct,...). Finally, a small text file is written to have
        some information on how the submission file was created.
        """
        # run parent process
        BatchProcess.run(self)
        print()

        # read in the feather file and then write the dataframe to csv (without index!)
        feather_submission = pd.read_feather(
            self.specified_odp(
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}_'
                         f'num_{self.num_similar}/mixed_prediction.feather'
            )
        )
        feather_submission.to_csv(
            self.specified_odp(
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}_'
                         f'num_{self.num_similar}/mixed_prediction.csv'
            ),
            index=False
        )

        # check the submission file
        check_submission(
            filename=self.specified_odp(
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}_'
                         f'num_{self.num_similar}/mixed_prediction.csv'
            ),
            nr_customers=self.nr_rows
        )

        # write a small description txt file
        file = open(self.specified_odp(
            filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}'
                     f'_num_{self.num_similar}'
                     f'/description.txt'
        ), 'w')
        file.write(f'Similarities: {self.similarity_version}\n')
        file.write(f'Number of similar items / history item: {self.num_similar}\n')
        file.write(f'Method: recency/mixed')
        file.close()

    def df_apply(self, hist):
        """
        Method to find the recommendations for a single customer.
        :param hist: a row of the self.customers_transactions_df dataframe
        :return: a string with 12 recommendations
        """
        customer_purchase_hist = hist['purchase_history']
        customer_default = hist['most_popular']

        # remove all duplicates and all article_ids that are not linked to an image
        customer_purchase_hist = list(dict.fromkeys([
            c for c in article_str_to_list(article_id_str=customer_purchase_hist) if c not in self.no_img_articles
        ]))

        # truncate such that only the self.num_similar last purchased articles are left
        customer_purchase_hist = customer_purchase_hist[:self.num_similar]

        # remove all '' (if empty history, '' can be in customer_purchase_hist)
        customer_purchase_hist = [x for x in customer_purchase_hist if len(x) > 0]
        if len(customer_purchase_hist) == 0:
            # now, if the history is empty, just return the default recommendations (popularity/age based)
            return customer_default

        most_similar_article_ids = [
            self.similarity_ids_df.loc[article_id, '1']
            for article_id in customer_purchase_hist
        ] + article_str_to_list(customer_default)

        # return the 12 most similar articles
        return article_list_to_str(most_similar_article_ids[:12])

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. The goal is to find 12 recommendations for each customer included in the
        current batch.
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        # obtain the relevant part of the dataframe
        part_of_df = self.customers_transactions_df.iloc[min_row_ind:min_row_ind + self.nr_rows_per_batch].copy()

        # get the prediction for each customer in this batch
        part_of_df['prediction'] = part_of_df.apply(self.df_apply, axis=1)

        # filter to only have the relevant columns before writing to feather
        part_of_df = part_of_df[['customer_id', 'prediction']].copy()
        part_of_df = part_of_df.reset_index()
        part_of_df.to_feather(
            self.specified_odp(
                filename=f'mixed_prediction_{batch_nr + 1}.feather',
                creation=True
            )
        )
        del part_of_df


if __name__ == '__main__':
    """
    usage: e.g. "python recommendations_mixed_recency.py --similarity_version similarities_ResNet50_W128_H128 
                --nr_rows_per_batch 10000 --num_similar 3"
    """
    arguments = sys.argv[1:]

    submission_calculator = RecencyMixedRecommender(arguments)
    submission_calculator.run()
