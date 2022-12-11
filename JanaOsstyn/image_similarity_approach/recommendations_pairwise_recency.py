import os

import pandas as pd

from batch_process import BatchProcess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

from functions import *


class RecencyPairwiseRecommender(BatchProcess):
    """
    Subclass of BatchProcess.
    Predict articles for each customer.
    The 12 most recent purchases are extracted for each customer. Then, for each of these 12, the ith most similar
    article is taken as recommendation. If a customer has less than 12 purchases, the recommendations are complemented
    with the most popular articles of the group of people with approximately the same age. See preprocessing.ipynb for
    more information on this.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        """
        Constructor
        :param args: a list of arguments that will be processed by self.read_arguments.
        """
        BatchProcess.__init__(self, modulename='recency_pairwise_prediction')

        self.sim_level = None                                       # similarity level (1 = most similar)
        self.similarity_version = None                              # the version to use (similarity depends on emb.)
        self.similarity_df = None                                   # the dataframe containing all similarities
        self.cold_start_recommendations_df = pd.read_feather(       # a dataframe with cold start recommendations
            odp(filename='cold_start_recommendations.feather')
        )
        self.customer_transaction_df = pd.read_feather(             # a dataframe with customer purchase history
            idp(filename='customer_transactions_processed.feather')
        )
        self.nr_rows = self.customer_transaction_df.shape[0]        # nr rows = nr of customers to find predictions for
        self.recommending_dict = None                               # dict with the similar items def. by self.sim_level
        self.base_filenames = ['recency_pairwise_prediction']       # names for final and temporary files

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
        print('python recommendations_pairwise_recency.py followed by:')
        print('\t--similarity_version value (required)')
        print('\t--sim_level value (optional, default 1)')
        print('\t--nr_rows_per_batch value (optional, default 10000)')
        exit()

    def read_arguments(self, args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        The arguments should include:
            - '--similarity_version' followed by the name of a directory containing similarity feather files (required)
            - '--sim_level' followed by an int, indicates how similar the recommendations are (optional, default 1)
            - '--nr_rows_per_batch' followed by the batch size (number of rows to be considered per batch)
        If 'help' is included in args or an unknown argument is found, self.print_help is called.
        The value of self.output_directory is based on the model name, and the image width and height.
        :param args: a list of arguments
        """
        if 'help' in args:
            self.print_help()

        self.nr_rows_per_batch = 10000  # default 10000
        self.sim_level = 1              # default 1
        for i, arg in enumerate(args):
            if arg == '--sim_level':
                self.sim_level = int(arguments[i + 1])
            elif arg == '--nr_rows_per_batch':
                self.nr_rows_per_batch = int(arguments[i + 1])
            elif arg == '--similarity_version':
                self.similarity_version = arguments[i + 1]
            elif not arg.startswith('--'):
                continue
            else:
                print(f'Unknown argument {arg}, expected:')
                self.print_help()

        # given the arguments, we can now extract useful entities
        self.similarity_df = pd.read_feather(odp(f'similarities/{self.similarity_version}/similarities_ids.feather'))
        recommendation_list = self.similarity_df[str(self.sim_level)].values.tolist()
        article_id_list = self.similarity_df['article_id'].values.tolist()
        self.recommending_dict = {article_id_list[i]: recommendation_list[i] for i in range(len(article_id_list))}
        self.output_directory = f"{self.modulename}_{self.similarity_version.replace('similarities_', '')}" \
                                f"_sim_{self.sim_level}"

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method forms an extension on the self.can_run of the superclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.similarity_df is not None and \
            self.recommending_dict is not None and \
            self.similarity_version is not None

    # ------------------------------------------------------------------------------------------------------------------
    # pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

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
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}'
                         f'_sim_{self.sim_level}/recency_pairwise_prediction.feather'
            )
        )
        feather_submission.to_csv(
            self.specified_odp(
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}'
                         f'_sim_{self.sim_level}/recency_pairwise_prediction.csv'
            ),
            index=False
        )

        # check the submission file
        check_submission(
            filename=self.specified_odp(
                filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}'
                         f'_sim_{self.sim_level}/recency_pairwise_prediction.csv'
            ),
            nr_customers=self.nr_rows
        )

        # write a small description txt file
        file = open(self.specified_odp(
            filename=f'{self.modulename}_{self.similarity_version.replace("similarities_", "")}'
                     f'_sim_{self.sim_level}/description.txt'
        ), 'w')
        file.write(f'Similarities: {self.similarity_version}\n')
        file.write(f'Sim level: {self.sim_level}\n')
        file.write(f'Method: recency/pairwise')
        file.close()

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. The goal is to find 12 recommendations for each customer included in the
        current batch.
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        # obtain the relevant part of the dataframe
        max_row_ind = min(min_row_ind + self.nr_rows_per_batch, self.nr_rows)
        part_of_df = self.customer_transaction_df.iloc[min_row_ind:max_row_ind].copy()

        # split the string of article_ids such that each article_id has its own column (and only keep 12 most recent)
        try:
            history_df = part_of_df['purchase_history'].str.split(pat=' ', expand=True)[[i for i in range(12)]]
        except KeyError:  # there are no 12 columns
            history_df = part_of_df['purchase_history'].str.split(pat=' ', expand=True)
            while 11 not in history_df.columns:
                history_df[history_df.columns[-1] + 1] = ''

        # add customer_id and age to the dataframe
        history_df.insert(0, 'customer_id', part_of_df['customer_id'])
        history_df.insert(1, 'age', part_of_df['age'])

        # replace each article_id by the article_id of the most similar article
        for column in range(12):
            history_df[column] = history_df[column].apply(
                lambda x: 'None' if x is None or x == '' else self.recommending_dict[x]
            )

        # now merge with the cold start recommendations such that each customer will have 12 recommendations, even if
        # he/she didn't purchase 12 or more articles
        history_df = history_df.merge(self.cold_start_recommendations_df, how='left', on='age')
        history_df = history_df.drop(columns=['age'])

        # everything is ready, we just have to convert everything to lists, remove the 'None' values and truncate to 12
        prediction = history_df.values.tolist()
        prediction = [[x for x in h if x != 'None'] for h in prediction]
        prediction = [(h[0], ' '.join(h[1:13])) for h in prediction]  # 0 = customer_id, other indices = article_ids

        # convert to a dataframe and write to a feather file
        prediction_df = pd.DataFrame.from_records(prediction, columns=['customer_id', 'prediction'])
        prediction_df.to_feather(
            self.specified_odp(
                filename=f'recency_pairwise_prediction_{batch_nr + 1}.feather',
                creation=True
            )
        )
        del part_of_df
        del history_df
        del prediction
        del prediction_df


if __name__ == '__main__':
    # TODO!
    arguments = sys.argv[1:]

    model_names = [
        'ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'InceptionV3',
        'InceptionResNetV2', 'VGG16', 'VGG19', 'Xception'
    ]

    k = 1
    for s in range(6, 11):
        for x in ['', 'extended_']:
            for model_name in model_names:
                print(f'{k}/{len(model_names) * 10}', model_name)
                arguments = [
                    '--similarity_version', f'{x}similarities_{model_name}_W128_H128', '--sim_level', str(s),
                    '--nr_rows_per_batch', '10000'
                ]

                submission_calculator = RecencyPairwiseRecommender(arguments)
                submission_calculator.run()
                k += 1
