from sklearn import preprocessing
import pandas as pd

class MultiLabelEncoder:
    """
    Object that holds all encoders so that they can be called upon at any time.
    """

    def __init__(self):
        self.encoders = {}
        self.decoders = {}  # holds the translation dicts

    def create_encoder(self, name):
        """
        Create a new encoder, make sure that this name is unique to avoid problems
        """
        new_encoder = preprocessing.LabelEncoder()
        self.encoders[name] = new_encoder

    def encode(self, encoder_name, label, dataframe):
        """
        Label encode a column in a dataframe.
        : encoder_name: name of the encoder that is to be used
        : label: name of column
        : dataframe: the actual dataframe

        """
        # create decoder dict and store original
        decoder = pd.DataFrame(columns=['original', 'encoded'])
        decoder["original"] = dataframe[label]

        # call encoder
        encoder = self.encoders[encoder_name]
        dataframe[label] = encoder.fit_transform(dataframe[label])

        # store encoded
        decoder["encoded"] = dataframe[label]
        self.decoders[encoder_name] = decoder

        return dataframe

    def decode_df(self, encoder_name, label, dataframe):
        """
        Decode a label encoded dataframe column
        : encoder_name: name of the encoder that was used
        : label: name of column
        : dataframe: the actual dataframe containing the encoded column
        """
        decoder = self.decoders[encoder_name]
        dataframe.rename(columns={label: 'encoded'}, inplace=True)
        dataframe = dataframe.merge(decoder, how="left", on="encoded")
        dataframe.drop(columns={"encoded"}, inplace=True)
        dataframe.rename(columns={"original": label}, inplace=True)
        return dataframe