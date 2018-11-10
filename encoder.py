from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Dense
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import pickle
# import pdb    

len_sentence = 100

def apply_softmax(arr, t=1.0):
    arr = np.array(arr)
    e = np.exp( arr/t)
    dist = e / np.sum(e)
    return dist

def apply_attention(arr, att):
    att = att.reshape(10,1)
    arr = np.multiply(att, arr)
    fin = np.sum(arr, axis=0)
    fin = fin.reshape(1,1,len_sentence)
    return fin

def getcorrectOrder(para, labels):
    print("******************************")
    print(labels, labels.shape)
    print("******************************")
    arr1inds = labels.argsort()
    # sorted_arr1 = labels[arr1inds[::-1]]
    sorted_arr2 = para[arr1inds[::-1]]
    n = para.shape[0]
    y = np.zeros((n,n))
    for i in range(n):
        # print(i)
        # print(i, labels[0][i])
        y[labels[0][i]][i] = 1
    return sorted_arr2, y


if __name__ == '__main__':
    # start = np.ones((1,100))
    inputs1 = Input(shape=(1,len_sentence))
    # inputs1 = tf.convert_to_tensor(start, np.float32)
    #encoder
    lstm1, state_h, state_c = LSTM(10, return_sequences=True, return_state=True)(inputs1)
    model_lstm = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

    #decoder
    model = Sequential()
    model.add(LSTM(10, input_shape=(10,len_sentence), dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
    # model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    data = np.ones(((3,10,len_sentence)))
    warr, uarr, barr = 1, 1, 1
    weightLSTM = model_lstm.layers[1].get_weights()
    for para in data:
        n = len(para)
        # encoder starts
        # set weights of encoder from decoder
        model_lstm.layers[1].set_weights(weightLSTM)
        # first iteration for start
        start = np.zeros((1, 1, len_sentence))
        
        # start_tf = tf.convert_to_tensor(start, np.float32)
        # start_tf = tf.reshape(start_tf, shape=[inputs1.shape[0], inputs1.shape[1], inputs1.shape[2]])
        # print(start_tf.shape, start_tf)
        lstm_output, hidden_states, cell_states = model_lstm.predict(start)
        att = apply_softmax(lstm_output)
        par_att = apply_attention(para, att)
        # par_att_tf = tf.convert_to_tensor(par_att, np.float32)
        m = 100
        for i in range(m):
            model_lstm.layers[1].states[0] = hidden_states
            model_lstm.layers[1].states[1] = cell_states
            # lstm out
            lstm_output, hidden_states, cell_states = model_lstm.predict(par_att)
            # softmax on lstm out
            att = apply_softmax(lstm_output)
            # apply attention on sentence
            par_att = apply_attention(para, att)
            # # create tensor for input
            # par_att_tf = tf.convert_to_tensor(par_att, np.float32)
            
        # decoder starts
        model.layers[0].states[0] = hidden_states
        model.layers[0].states[1] = cell_states
        # construct input and output for decoder
        
        labels = np.arange(n).reshape(1,n)
        np.random.shuffle(labels)
        xtrain, ytrain = getcorrectOrder(para, labels)
        # print(xtrain.shape)
        # xtrain = xtrain[0:-1,:]
        start = np.zeros(len_sentence).reshape(1,len_sentence)
        # print(start.shape, xtrain.shape)
        # xtrain = np.concatenate((start, xtrain), axis = 1)

        # xtrain.reshape(10,1,100)
        
        # train decoder
        print(model.summary())
        print(xtrain.shape)
        # ytrain = np.argwhere(ytrain)[:,1]
        ytrain = ytrain.reshape(1,10,10)
        print(ytrain.shape)
        model.fit(xtrain, ytrain, batch_size =1, epochs = 300,  verbose = 0)
        weightLSTM = model_lstm.layers[1].get_weights()
        print("weight = ")
        print(weightLSTM)
        raw_input()
        # pdb.set_trace()

        # warr,uarr, barr = weightLSTM

