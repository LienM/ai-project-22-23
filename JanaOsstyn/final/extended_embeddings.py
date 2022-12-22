import os
import datetime

from batch_process import BatchProcess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import spacy

from functions import *


class ExtendedEmbeddingCalculator(BatchProcess):
    """
    Subclass of BatchProcess.
    Allows calculation of extended embeddings (image embedding + word embedding of article description).
    This class assumes the embeddings are already created with embeddings.py. It calculates the word2vec embeddings
    for each article description and concatenates these to all image embeddings calculated in embeddings.py.
    If e.g. the image embeddings were created for 3 models, then the extended embeddings will be written to file for
    these 3 embeddings.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        BatchProcess.__init__(self, modulename='extended_embeddings')

        self.model = None                               # pretrained model name of original embeddings
        self.article_df = pd.read_feather(              # dataframe containing all article and their properties
            idp(filename='articles_processed.feather')
        )
        self.nr_rows = self.article_df.shape[0]         # nr of rows in the article frame (= # articles)
        self.base_filenames = ['extended_embeddings']   # subf. extended_embeddings_X.feather, final embeddings.feather
        self.word2vec_shape = 300                       # shape of the word2vec embeddings (fixed 300)
        self.check_column = ['article_id'] + self.article_df['article_id'].values.tolist()

        self.read_arguments(args)                       # read the arguments
        self.fetch_w2v_model()                          # fetch the pretrained word2vec model

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
        print('python extended_embeddings.py followed by:')
        print('\t--nr_rows_per_batch value (optional, default 1000)')
        exit()

    def read_arguments(self, args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        The arguments may include:
            - '--nr_rows_per_batch' followed by the batch size (number of rows to be considered per batch)
        If 'help' is included in args or an unknown argument is found, self.print_help is called.
        The value of self.output_directory is not relevant for this class as the output is based on existing directories
        with embeddings.
        :param args: a list of arguments
        """
        if 'help' in args:
            self.print_help()

        self.nr_rows_per_batch = 1000
        for i, arg in enumerate(args):
            if arg == '--nr_rows_per_batch':
                self.nr_rows_per_batch = int(arguments[i + 1])
            elif not arg.startswith('--'):
                continue
            else:
                print(f'Unknown argument {arg}, expected:')
                self.print_help()

        # not relevant
        self.output_directory = ''

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method forms an extension on the self.can_run of the superclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.model is not None and \
            super().can_run()

    # ------------------------------------------------------------------------------------------------------------------
    # pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

    def fetch_w2v_model(self):
        """
        Fetch the word2vec model (en_core_web_md).
        If it is not stored locally, then load it from origin.
        """
        try:
            self.model = spacy.load('en_core_web_md')
        except Exception as e:
            os.system('python -m spacy download en_core_web_md')
            self.model = spacy.load('en_core_web_md')

    def run(self):
        """
        Run the process. This is done in batches of self.nr_rows_per_batch lines each.
        Progress is output to the terminal. Time is kept and reported such that one can see how long the run took.
        Subfiles are written to a 'creation' output directory. All files in this directory are then joined into one or
        more single files after which the 'creation' directory is removed to save storage space.
        The final file(s) is/are written to a directory specified by the subclass. It is mostly a name related to the
        modulename and to the model used to generate the embeddings.
        This method overrides the method self.run of the superclass as execution is not entirely as the self.run of
        the superclass.
        """
        if not self.can_run():
            print('One or more arguments are not provided or could not be derived!')
            exit()

        start = time.time()

        # remove directory to not rely on potential earlier generated files
        # e.g. if in a previous run 100 files were generated, and this run will generate only 90 files, then 90 files
        # will probably be overwritten while the 10 remaining files are joined with the 90 files in a later phase
        remove_output_directory_sequence_if_exists(f'{self.modulename}_creation')

        # execute the loop over all batches
        self.loop_function()

        # join all files
        for base_filename in self.base_filenames:
            print(base_filename)
            self.join_files(base_filename=base_filename)

        # remove the temporary 'creation' output directory and create a new directory for the joined file(s)
        remove_output_directory_sequence_if_exists(base_directory=f'{self.modulename}_creation')

        # concatenate the new embeddings with the image embeddings
        self.concat_with_image_embeddings()

        # remove any other temporary results
        remove_file_if_exists(filename=self.specified_odp(filename='embeddings.feather'))
        remove_output_directory_sequence_if_exists(base_directory=f'{self.modulename}')

        end = time.time()
        print('Total elapsed time:', str(datetime.timedelta(seconds=end - start)))

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. The goal is to calculate the word embeddings for all rows within the
        specified batch. Concatenation with existing image embeddings happens in a later phase in the
        self.concat_with_image_embeddings method
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        # obtain the relevant part of the dataframe
        part_of_df = self.article_df.iloc[min_row_ind:min(min_row_ind + self.nr_rows_per_batch, self.nr_rows)]
        description_series = part_of_df['detail_desc'].fillna('')

        # create the dataframe with the embeddings and write to file
        embedding_df = pd.DataFrame(
            [x.vector for x in self.model.pipe(description_series)],
            columns=[str(j) for j in range(self.word2vec_shape)]
        )
        embeddings_df = embedding_df.reset_index()
        embeddings_df['article_id'] = part_of_df['article_id']
        embeddings_df.to_feather(
            self.specified_odp(
                filename=f'extended_embeddings_{batch_nr + 1}.feather',
                creation=True
            )
        )
        del part_of_df

    def concat_with_image_embeddings(self):
        """
        Concatenate the image embeddings with the newly created word2vec embeddings
        """
        # read the word2vec embeddings
        embedding_df = pd.read_feather(self.specified_odp(filename='extended_embeddings.feather'))

        # for each of the image embeddings, extend the embeddings with the word2vec embeddings
        content = list_directory_if_exists(odp('embeddings'))
        for directory in content:
            if directory.startswith('extended'):
                continue
            # read the image embeddings
            img_embedding_df = pd.read_feather(odp(f'embeddings/{directory}/embeddings.feather'))

            # find the maximum column to update the embedding column names before attaching it to the embeddings
            max_column = int(img_embedding_df.columns.values.tolist()[-1])
            embedding_df.columns = [str(max_column + 1 + j) for j in range(self.word2vec_shape)]

            # concatenate both dataframes (horizontally)
            new_embedding_df = img_embedding_df.merge(embedding_df, on='article_id', how='left')
            new_embedding_df.columns = img_embedding_df.columns + embedding_df.columns

            # check
            assert sorted(new_embedding_df['article_id'].values.tolist()) == \
                   sorted(img_embedding_df['article_id'].values.tolist)

            # write the dataframe to a feather file
            create_directory_if_not_exists(directory=odp(f'embeddings/extended_{directory}'))
            img_embedding_df.to_feather(odp(f'embeddings/extended_{directory}/embeddings.feather'))
            del img_embedding_df


if __name__ == '__main__':
    # e.g. python embeddings.py --nr_rows_per_batch 2000
    arguments = sys.argv[1:]

    extended_embedding_calculator = ExtendedEmbeddingCalculator(arguments)
    extended_embedding_calculator.run()
