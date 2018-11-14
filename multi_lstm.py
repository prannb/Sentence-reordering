import numpy as np
import random as ran
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Lambda, TimeDistributed, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split

epochs = 30
max_sentence = 10
sen_len = 300

def build_model():
    model = Sequential()
    model.add(LSTM(50,input_shape = (max_sentence,sen_len),return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(max_sentence, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
            optimizer=Adam(),
            metrics=['accuracy'])
    return model

def create_new_y(y):
    n = y.shape[0]
    ny = np.zeros((max_sentence, n, max_sentence))
    i = 0
    while (i<n):
        req = y[i]
        for j in range(max_sentence):
            if (len(np.where(req==j)[0]) > 0):
                idx = np.where(req==j)[0][0]
                ny[j][i][idx] = 1
        i += 1
    return ny

if __name__ == "__main__":
    print("hello world")
    data = pickle.load(open('X_10_wtv_stories_replaced.pkl','rb'))
    Y = pickle.load(open('order_10_wtv_stories_replaced.pkl','rb'))
    data = data
    Y = Y

    # print(Y[0])
    # print(Y[0][0])
    # print(data[0])
    # print(data[0][0])


    data, X_test, Y, y_test = train_test_split(data, Y, test_size=0.2)
    print(Y.shape)
    print(data.shape)
    print(Y[1])
    print(data[1])

    exit()
    Y = create_new_y(Y)
    y_test = create_new_y(y_test)
    print(Y.shape)
    print(Y[0][0])
    models = []
    for i in range(max_sentence):
        models.append(build_model())
    weights = ''
    for i in range(max_sentence):
        if (i!=0):
            models[i].layers[0].set_weights(weights)
        print(i,"i********************************************i", i)
        models[i].fit(data, Y[i],
            batch_size=64,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test[i]))
        print(i,"i********************************************i", i)
        weights = models[i].layers[0].get_weights()