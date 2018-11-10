from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Sequential
import numpy as np
import tensorflow as tf
# define model

# inputs1 = Input(shape=(1, 1))
# print(inputs1)
n = 4
m = 10
input_entry = tf.random_gamma([n,1], 0.5)
inputs1 = Input(shape=(4,1))
input_entry = [np.array([1,2,3,4]).reshape((1,4,1))]
for x in range(9):
    input_entry.append(np.array([x,2,3,4]).reshape((1,4,1)))
# train = np.ones((3,2,1))
# test = np.zeros((3,1))
# lstm, hidden_state, cell_state = LSTM(1, input_shape = (2,1), return_sequences=True, return_state=True)(inputs1)
# model = Sequential()
# model.add(LSTM(1, input_shape = (2,1)))
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
# print(model.summary())
# model.fit(train, test)

lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
model_lstm = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])


for i in range(m):
    lstn_output, hidden_states, cell_states = model_lstm.predict(input_entry[i])
    model_lstm.layers[1].states[0] = hidden_states
    model_lstm.layers[1].states[1] = cell_states 
    print(lstn_output)
