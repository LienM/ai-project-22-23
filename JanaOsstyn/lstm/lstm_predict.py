import tensorflow as tf
import torch.cuda
from tensorflow import keras
from utils import *
import os


"""
Make predictions for each set of users (more than 3 purchases & less than 3 purchases)
"""

if not os.path.isdir('../../data/output'):
    os.mkdir('../../data/output')

for filename in ['../../data/transactions_gte_3_articles.csv', '../../data/transactions_lt_3_articles.csv']:

    # create sequences
    transactions = pd.read_csv(filename)
    sequences = padded_sequences(filename=filename)

    torch.cuda.is_available()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU')) < 1:
        exit()

    model = keras.models.load_model('lstm_model')

    padded_sequences = padded_sequences[:, 1:]
    number_of_customers = padded_sequences.shape[0]
    batch_size = 1000
    i = 0
    file_id = 'gte_3' if 'gte_3' in filename else 'lt_3'
    while True:
        min_val = batch_size * i
        max_val = min(batch_size * (i + 1), padded_sequences.shape[0])
        test_input = padded_sequences[min_val:max_val, :]
        test_users = np.array(transactions[min_val:max_val]['customer_id'])
        print(i, min_val, '-->', max_val, f'({(100 * max_val) / (transactions.shape[0])}%)')
        output_df = pd.DataFrame(test_users, columns=['customer_id'], index=[i for i in range(test_users.shape[0])])

        predictions = model.predict(test_input, batch_size=1)
        for j in range(12):
            recommendations = [np.argmax(predictions[u]) for u in range(test_users.shape[0])]
            output_df[f'recommendation_{j + 1}'] = recommendations
            for k in range(len(recommendations)):
                predictions[k][recommendations[k]] = -1
        output_df.to_csv(f'output/output_{i}_{file_id}.csv', index=False)

        i += 1

        if max_val == number_of_customers:
            break
