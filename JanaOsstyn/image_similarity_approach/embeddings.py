import os

import pandas

from batch_process import BatchProcess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

from keras import Sequential
from keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionV3, \
    InceptionResNetV2, VGG16, VGG19, Xception
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.layers import GlobalMaxPooling2D
from keras.utils import img_to_array
from keras.utils import load_img as keras_load_img

from functions import *


class EmbeddingCalculator(BatchProcess):
    """
    Subclass of BatchProcess.
    Allows embedding calculation in batches of rows
    """

    # ------------------------------------------------------------------------------------------------------------------
    # constructor, getters & setters
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, args):
        """
        Constructor
        :param args: a list of arguments that will be processed by self.read_arguments.
        """
        BatchProcess.__init__(self, modulename='embeddings')

        self.img_width, self.img_height = 128, 128  # width and height of the images
        self.model_name = None                      # name of the model for embedding calculation
        self.model = None                           # the loaded model
        self.preprocess_input_function = None       # each model requires a separate preprocessing function
        self.embedding_shape = None                 # the shape of the embedding (how many values it has)
        self.article_df = pd.read_feather(idp(      # the dataframe with the article information
            filename='articles_processed.feather'
        ))
        self.nr_rows = self.article_df.shape[0]     # the number of rows in the article dataframe (needed by superclass)
        self.base_filenames = ['embeddings']        # subfiles are embeddings_X.feather, final file embeddings.feather

        self.read_arguments(args)                   # read the arguments
        self.fetch_model()                          # fetch the model specified in the arguments

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
        print('python embeddings.py followed by:')
        print('\t--model model_name (required)')
        print('\t--nr_rows_per_batch value (optional, default 1000)')
        exit()

    def read_arguments(self, args):
        """
        Method to read and interpret the arguments provided by the user.
        This method may call the self.print_help method as described in the documentation of self.print_help.
        The arguments may include:
            - '--model' followed by a name of a pretrained model to be loaded from Keras (required)
            - '--nr_rows_per_batch' followed by the batch size (number of rows to be considered per batch, default 1000)
        If 'help' is included in args or an unknown argument is found, self.print_help is called.
        The value of self.output_directory is based on the model name, and the image width and height.
        :param args: a list of arguments
        """
        if 'help' in args:
            self.print_help()

        self.nr_rows_per_batch = 1000
        for i, arg in enumerate(args):
            if arg == '--model':
                self.model_name = arguments[i + 1]
            elif arg == '--nr_rows_per_batch':
                self.nr_rows_per_batch = int(arguments[i + 1])
            elif not arg.startswith('--'):
                continue
            else:
                print(f'Unknown argument {arg}, expected:')
                self.print_help()

        # given the arguments, we can define an output directory related to the input parameters
        self.output_directory = f'embeddings_{self.model_name}_W{self.img_width}_H{self.img_height}'

    def can_run(self):
        """
        Quick check whether all preconditions (in terms of arguments) are met before run is executed.
        This method forms an extension on the self.can_run of the superclass.
        :return: True if all preconditions are met, False otherwise
        """
        return \
            self.model is not None and \
            self.preprocess_input_function is not None and \
            self.embedding_shape is not None and \
            super().can_run()

    # ------------------------------------------------------------------------------------------------------------------
    # pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

    def fetch_model(self):
        """
        Fetch the appropriate model as specified by the user. There is a list of 11 possible models to choose from.
        An exception will be raised if the specified model is not one of these 11 models.
        """
        # https://keras.io/api/applications/
        if self.model_name == 'ResNet50':  # 2048, 23 587 712 parameters
            base_model = ResNet50(weights='imagenet', include_top=False,
                                  input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'ResNet50V2':  # 2048, 23 564 800 parameters
            base_model = ResNet50V2(weights='imagenet', include_top=False,
                                    input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'ResNet101':  # 2048, 42 658 176 parameters
            base_model = ResNet101(weights='imagenet', include_top=False,
                                   input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'ResNet101V2':  # 2048, 42 626 560 parameters
            base_model = ResNet101V2(weights='imagenet', include_top=False,
                                     input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'ResNet152':  # 2048, 58 370 944 parameters
            base_model = ResNet152(weights='imagenet', include_top=False,
                                   input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'ResNet152V2':  # 2048, 58 331 648 parameters
            base_model = ResNet152V2(weights='imagenet', include_top=False,
                                     input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = resnet_preprocess_input
        elif self.model_name == 'InceptionV3':  # 2048, 21 802 784 parameters
            base_model = InceptionV3(weights='imagenet', include_top=False,
                                     input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = inception_preprocess_input
        elif self.model_name == 'InceptionResNetV2':  # 1536, 54 336 736 parameters
            base_model = InceptionResNetV2(weights='imagenet', include_top=False,
                                           input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = inception_resnet_v2_preprocess_input
        elif self.model_name == 'VGG16':  # 512, 14 714 688
            base_model = VGG16(weights='imagenet', include_top=False,
                               input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = vgg16_preprocess_input
        elif self.model_name == 'VGG19':  # 512, 20 024 384
            base_model = VGG19(weights='imagenet', include_top=False,
                               input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = vgg19_preprocess_input
        elif self.model_name == 'Xception':  # 2048, 20 861 480 parameters
            base_model = Xception(weights='imagenet', include_top=False,
                                  input_shape=(self.img_width, self.img_height, 3))
            self.preprocess_input_function = xception_preprocess_input
        else:
            raise Exception('Model not recognized')

        base_model.trainable = False
        self.model = Sequential([base_model, GlobalMaxPooling2D()])

        print(self.model.summary())
        self.embedding_shape = self.model.get_layer(self.model.layers[1].name).output_shape[1]

    def batch_function(self, min_row_ind, batch_nr):
        """
        This method will be applied to a single batch, ranging from the row with index min_row_ind to the row with
        min_row_ind + self.nr_rows_per_batch. The goal is to calculate the embeddings for all rows within the specified
        batch.
        :param min_row_ind: the index of the lower bound of the rows treated in this batch
        :param batch_nr: the number of the batch (used for filename)
        """
        # obtain the relevant part of the dataframe
        part_of_df = self.article_df.iloc[min_row_ind:min(min_row_ind + self.nr_rows_per_batch, self.nr_rows)]
        img_df = pandas.DataFrame(part_of_df['image_name'])

        # split in images that exist and images that do not exist
        non_existing_images_df = img_df[img_df['image_name'] == 'does not exist'].copy()
        existing_images_df = img_df[img_df['image_name'] != 'does not exist'].copy()

        # generate predictions for the existing images
        existing_images_df['img'] = existing_images_df['image_name'].apply(
            lambda x: keras_load_img(idp(f'images/{x}'), target_size=(self.img_width, self.img_height))
        )
        existing_images_df['img_array'] = existing_images_df['img'].apply(
            lambda x: self.preprocess_input_function(np.expand_dims(img_to_array(x), 0))
        )
        prediction = self.model.predict(np.vstack(existing_images_df['img_array']), batch_size=512, verbose=0)
        existing_images_prediction_df = pd.DataFrame(
            prediction,
            columns=[str(j) for j in range(self.embedding_shape)]
        )

        # create an equivalent dataframe for the non-existing images (embeddings with only zeros)
        non_existing_images_prediction_df = pd.DataFrame(
            np.zeros((non_existing_images_df.shape[0], self.embedding_shape)),
            columns=[str(j) for j in range(self.embedding_shape)]
        )

        # set index for both frames
        existing_images_prediction_df = existing_images_prediction_df.set_index(existing_images_df.index)
        non_existing_images_prediction_df = non_existing_images_prediction_df.set_index(non_existing_images_df.index)

        # concatenate frames to have a single frame with an embedding for all users and write to file
        embeddings_df = pd.concat([existing_images_prediction_df, non_existing_images_prediction_df])
        embeddings_df = embeddings_df.reset_index()
        embeddings_df.to_feather(
            self.specified_odp(
                filename=f'embeddings_{batch_nr + 1}.feather',
                creation=True
            )
        )
        del part_of_df
        return


if __name__ == '__main__':
    arguments = sys.argv[1:]

    embedding_calculator = EmbeddingCalculator(arguments)
    embedding_calculator.run()
