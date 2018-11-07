from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
import numpy as np


# TODO: import the training data
train = []
test = []

# initialise the model
lstm,  = LSTM(1, return_sequences=True, return_state=True)(inputs1)


# training starts here
for paragraph in train:
    for sentence in paragraph:
