import datetime
import gc
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

from sklearn.metrics.pairwise import cosine_similarity

from functions import *
from batch_process import BatchProcess


class SimilarityCalculator(BatchProcess):
    """
    Subclass of BatchProcess.
    Allows similarity calculation in batches of rows.
    Only the 250 most similar articles are kept for each article, as storing them all cannot be handled in a single
    dataframe which makes similarity_lookup in a later phase more difficult than needed.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        """
        Constructor
        :param args: a list of arguments that will be processed by self.read_arguments.
        """
        BatchProcess.__init__(self, modulename='similarities')

        self.embedding_version = None                               # version of embeddings to use
        self.embedding_df = None                                    # dataframe with embeddings
        self.base_filenames = [
            'similarities_idx',                                     # file for most similar article indices
            'similarities_values',                                  # file for most similar article values
            'similarities_ids'                                      # file for most similar article ids
        ]
        article_df = pd.read_feather(idp(
            filename='articles_processed.feather'
        ))
        article_ids = article_df['article_id'].values.tolist()
        self.article_dict = {                                       # a dictionary mapping indices to article ids
            i: article_id
            for i, article_id in enumerate(article_ids)
        }
        self.keep_nr = 250                                          # keep the 250 most similar articles only

        self.read_arguments(args)                                   # read the arguments

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
        print('python similarities.py followed by:')
        print('\t--embedding_version value (required)')
        print('\t--nr_rows_per_batch value (optional, default 1000)')
        exit()

    def read_arguments(self, args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        The arguments may include:
            - '--embedding_version' followed by the name of a directory containing an embeddings.feather file (required)
            - '--nr_rows_per_batch' followed by the batch size (number of rows to be considered per batch, default 1000)
        If 'help' is included in args or an unknown argument is found, self.print_help is called.
        The value of self.output_directory is based on the model name, and the image width and height.
        :param args: a list of arguments
        """
        if 'help' in args:
            self.print_help()

        self.nr_rows_per_batch = 1000  # default
        for i, arg in enumerate(arguments):
            if arg == '--embedding_version':
                self.embedding_version = arguments[i + 1]
            elif arg == '--nr_rows_per_batch':
                self.nr_rows_per_batch = int(arguments[i + 1])
            elif not arg.startswith('--'):
                continue
            else:
                print(f'Unknown argument {arg}, expected:')
                self.print_help()

        # given the arguments, we can now extract useful entities
        self.embedding_df = pd.read_feather(odp(filename=f'embeddings/{self.embedding_version}/embeddings.feather'))
        self.nr_rows = self.embedding_df.shape[0]
        self.output_directory = self.embedding_version.replace('embeddings', 'similarities')

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method forms an extension on the self.can_run of the superclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.embedding_version is not None and \
            self.embedding_df is not None and \
            super().can_run()

    # ------------------------------------------------------------------------------------------------------------------
    # pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. The goal is to calculate the pairwise similarities for all rows within
        the specified batch with respect to all rows. Thus, in case of batch size 1000, the similarities are calculated
        for each of these 1000 rows to each row in the whole dataframe.
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        # obtain the relevant part of the dataframe
        batch_article_ids = self.embedding_df['article_id']
        embed_df = self.embedding_df.drop(columns=['article_id'])
        part_of_embed_df = embed_df.iloc[min_row_ind:min_row_ind + self.nr_rows_per_batch]

        # calculates similarity
        similarities = cosine_similarity(
            part_of_embed_df,
            embed_df
        )

        # sort the similarities such that highest similarity comes first
        sorted_row_idx = np.argsort(similarities, axis=1)[:, similarities.shape[1] - self.keep_nr::][:, ::-1]
        sorted_row_values = similarities[np.arange(similarities.shape[0])[:, None], sorted_row_idx]

        # create dataframes from numpy similarity matrices
        similarity_idx_df = pd.DataFrame(sorted_row_idx)
        similarity_value_df = pd.DataFrame(sorted_row_values)

        similarity_idx_df.columns = similarity_idx_df.columns.astype(str)
        similarity_value_df.columns = similarity_value_df.columns.astype(str)

        similarity_idx_df.index = batch_article_ids[min_row_ind:min_row_ind + self.nr_rows_per_batch]
        similarity_value_df.index = batch_article_ids[min_row_ind:min_row_ind + self.nr_rows_per_batch]

        similarity_idx_df = similarity_idx_df.reset_index()
        similarity_value_df = similarity_value_df.reset_index()

        # write dataframes to file
        similarity_idx_df.to_feather(
            self.specified_odp(
                filename=f'similarities_idx_{batch_nr + 1}.feather',
                creation=True
            )
        )
        similarity_value_df.to_feather(
            self.specified_odp(
                filename=f'similarities_values_{batch_nr + 1}.feather',
                creation=True
            )
        )

        # copy the idx dataframe and replace the indices by the corresponding ids
        similarity_ids_df = similarity_idx_df.copy()
        for column in range(self.keep_nr):
            similarity_ids_df[str(column)] = similarity_ids_df[str(column)].apply(
                lambda x: self.article_dict[x]
            )

        # write the ids to file
        similarity_ids_df.to_feather(
            self.specified_odp(
                filename=f'similarities_ids_{batch_nr + 1}.feather',
                creation=True
            )
        )

        del similarity_idx_df
        del similarity_value_df


if __name__ == '__main__':
    arguments = sys.argv[1:]

    similarity_calculator = SimilarityCalculator(arguments)
    similarity_calculator.run()
