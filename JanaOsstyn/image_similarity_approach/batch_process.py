import datetime
import gc
import math

from functions import *


class BatchProcess:
    """
    A base class for any process coded by me that is executed in batches.
    This class ensures a general workflow and a standardized way of reporting progress.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, modulename):
        """
        Constructor
        Anything except modulename is to be set by the subclass.
        :param modulename: name of the module (e.g. embeddings, similarities,...)
        """
        self.modulename = modulename            # modulename (used for names of output files)
        self.base_filenames = []                # one or more 'base' filenames (see self.join_files for more info)
        self.output_directory = None            # directory for output files
        self.__nr_rows_per_batch = None         # nr of rows each batch will have
        self.__nr_rows = None                   # total nr of rows
        self.__nr_batches = None                # nr of batches, calculated from self.nr_rows and self.nr_rows_per_batch

    @property
    def nr_rows_per_batch(self):  # getter
        return self.__nr_rows_per_batch

    @nr_rows_per_batch.setter
    def nr_rows_per_batch(self, value):  # setter
        self.__nr_rows_per_batch = value
        if self.nr_rows is not None and self.nr_rows_per_batch is not None:
            self.__nr_batches = math.ceil(self.nr_rows / self.nr_rows_per_batch)

    @property
    def nr_rows(self):  # getter
        return self.__nr_rows

    @nr_rows.setter
    def nr_rows(self, value):  # setter
        self.__nr_rows = value
        if self.nr_rows is not None and self.nr_rows_per_batch is not None:
            self.__nr_batches = math.ceil(self.nr_rows / self.nr_rows_per_batch)

    @property
    def nr_batches(self):  # getter -->  no setter as we want self.__nr_batches not to be changed from outside
        return self.__nr_batches

    # ------------------------------------------------------------------------------------------------------------------
    # helper methods
    # ------------------------------------------------------------------------------------------------------------------

    def print_help(self):
        """
        Prints some information for the user about the parameters that should be provided. This method is abstract and
        needs to be implemented by the subclass.
        This method is called if either:
            - the user passed 'help' as one of the arguments
            - the user provided a keyword that couldn't be recognized by the subclass
        """
        raise NotImplementedError

    def read_arguments(self,  args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        This method is abstract and needs to be implemented by the subclass.
        """
        raise NotImplementedError

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method is complemented with the preconditions of the subclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.nr_rows is not None and \
            self.nr_rows_per_batch is not None and \
            self.output_directory is not None

    def specified_odp(self, filename, creation=False):
        """
        This method is a shortcut to the desired output directory ('specified_odp' = 'specified_output_directory').
        It returns the path to the file with given filename. If the directory does not exist, it is created before the
        filename is returned.
        :param filename: the filename of the file
        :param creation: if True, the directory is '{self.modulename}_creation' instead of '{self.modulename}_creation'
        :return: the filename including the path to that file
        """
        directory = f'{self.modulename}_creation' if creation else self.modulename
        create_directory_if_not_exists(directory=odp(filename=directory))
        return odp(filename=f'{directory}/{filename}')

    # ------------------------------------------------------------------------------------------------------------------
    # pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        Run the process. This is done in batches of self.nr_rows_per_batch lines each.
        Progress is output to the terminal. Time is kept and reported such that one can see how long the run took.
        Subfiles are written to a 'creation' output directory. All files in this directory are then joined into one or
        more single files after which the 'creation' directory is removed to save storage space.
        The final file(s) is/are written to a directory specified by the subclass. It is mostly a name related to the
        modulename and to the model used to generate the embeddings.
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

        # join all groups of files
        # this may be a single file group, but it can be multiple too (such as values, ids,...)
        for base_filename in self.base_filenames:
            print(base_filename)
            self.join_files(base_filename=base_filename)

        # remove the temporary 'creation' output directory and create a new directory for the joined file(s)
        remove_output_directory_sequence_if_exists(f'{self.modulename}_creation')
        create_directory_if_not_exists(directory=self.specified_odp(filename=self.output_directory))

        # now move the joined files to their appropriate directory (the self.output_directory)
        move_files(
            filenames=[f'{base_filename}.feather' for base_filename in self.base_filenames],
            dir_from=self.specified_odp(filename='')[:-1],
            dir_to=self.specified_odp(filename=self.output_directory)
        )

        end = time.time()
        print('Total elapsed time:', str(datetime.timedelta(seconds=end - start)))

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. This method is to be implemented by the subclass.
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        raise NotImplementedError

    def loop_function(self):
        """
        This method loops over all batches and executes self.batch_function on each batch. It prints progress before
        each batch execution.
        """
        start = time.time()
        print(  # progress print before loop execution
            f"{current_time_stamp()} [>{' ' * self.nr_batches}] : 0.0%"
        )

        min_row_ind = 0
        for batch_nr in range(self.nr_batches):
            print(  # progress print during loop execution
                f"{current_time_stamp()} [{'=' * (batch_nr + 1)}>{' ' * (self.nr_batches - (batch_nr + 1))}] : "
                f"{round(100 * batch_nr * self.nr_rows_per_batch / self.nr_rows, 1)}%"
            )

            # execute the batch function, update min_row_ind and run gc.collect() for memory optimization
            self.batch_function(min_row_ind, batch_nr)
            min_row_ind += self.nr_rows_per_batch
            gc.collect()

        print(  # progress print after loop execution
            f"{current_time_stamp()} [{'=' * (self.nr_batches + 1)}] : "
            f"100.0%"
        )
        end = time.time()
        print('Elapsed time (loop):', str(datetime.timedelta(seconds=end - start)))
        return

    def join_files(self, base_filename):
        """
        One or more output files are created in each batch execution. These form fragments of the final file and need to
        be joined, which happens in this method. To know which files to join, base_filename gives an indication of all
        input files to collect and the name of the output file.
        :param base_filename: name of the base file
        """
        frame_list = []
        j = 1
        while True:
            try:
                # the filenames of the files to join are formed by base_filename with an additional number j
                # --> start by number 1 and increase j until no file with that number could be found anymore
                frame_list.append(pd.read_feather(
                    self.specified_odp(filename=f'{base_filename}_{j}.feather', creation=True))
                )
                j += 1
            except FileNotFoundError:
                break
        # concatenate frame
        complete_frame = pd.concat(frame_list, ignore_index=True)

        # remove columns that were potentially included in the batch_function but are not desirable
        if 'Unnamed: 0' in complete_frame.columns.values:
            complete_frame = complete_frame.drop(['Unnamed: 0'], axis=1)
        if 'index' in complete_frame.columns.values:
            complete_frame = complete_frame.drop(['index'], axis=1)

        # write the frame to a feather file and print the first lines as a sanity check
        complete_frame.to_feather(self.specified_odp(filename=f'{base_filename}.feather'))
        print(complete_frame.head())
        return
