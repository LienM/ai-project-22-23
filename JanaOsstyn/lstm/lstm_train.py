import matplotlib.pyplot as plt
import tensorflow as tf
import torch.cuda
from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense
from utils import *

"""
Create and train the LSTM model.
CAUTION! Training took 15 hours on my laptop <<don't try this at home>>!
Source: https://bond-kirill-alexandrovich.medium.com/lstm-for-real-time-recommendation-systems-f5191d564be5
"""

padded_sequences = pad_sequences('../../data/lstm/transactions_gte_3_articles.csv')

# split in x and y
x, y = padded_sequences[:, :-1], padded_sequences[:, -1]
print('Shape of training data:', x.shape)
print('Shape of labels:', y.shape)

# if no GPU available, quit (this stuff running on cpu only might not be a good idea...)
torch.cuda.is_available()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) < 1:
    exit()

# create model
model = Sequential()
model.add(Embedding(vocabulary_size(), 5, input_length=max_len() - 1))
model.add(Dropout(0.2))
model.add(LSTM(3))
model.add(Dropout(0.2))
model.add(Dense(vocabulary_size(), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fit and save model
history = model.fit(x, y, validation_split=0.3, verbose=1, epochs=5)
model.save('lstm_model')

# plot loss
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy
plt.plot(history.history['acc'], label='Train accuracy')
plt.plot(history.history['val_acc'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
