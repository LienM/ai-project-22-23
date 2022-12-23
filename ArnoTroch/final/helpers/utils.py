# utility functions to convert article_id and customer_id types
# source: https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635

DATA_PATH = '/Users/arnotroch/Documents/AIP/ai-project-22-23.nosync/data'

def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def article_id_str_to_int(series):
    return series.astype('int32')


def article_id_int_to_str(series):
    return '0' + series.astype('str')
