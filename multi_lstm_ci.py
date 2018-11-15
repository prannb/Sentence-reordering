import numpy as np
import random as ran
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Lambda, TimeDistributed, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
import copy

epochs = 25
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

def createfit(x, y, i):
    if (i == 0):
        n = y.shape[0]
        ny = np.zeros((n, max_sentence))
        # nY = np.zeros((y.shape[0], y.shape[1]))
        for k in range(n):
            idx = np.where(y[k]==i)[0]
            ny[k][idx[0]] = 1
        return (x, y, ny)
    else:
        n = y.shape[0]
        ny = np.zeros((n, max_sentence))
        nY = np.zeros((y.shape[0], y.shape[1]))
        nx = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
        for k in range(n):
            nx[k] = copy.deepcopy(x[k])
            nY[k] = copy.deepcopy(y[k])
            prev_idx = np.where(y[k]==(i-1))[0]
            if (len(prev_idx) == 0):
                continue
            else:
                pidx = prev_idx[0]
                nx[k][i-1] = copy.deepcopy(x[k][pidx])
                nx[k][pidx] = copy.deepcopy(x[k][i-1])
                nY[k][i-1] = copy.deepcopy(y[k][pidx])
                nY[k][pidx] = copy.deepcopy(y[k][i-1])
                idx = np.where(nY[k]==i)[0]
                if (len(idx) > 0):
                    ny[k][idx[0]] = 1
        return (nx, nY, ny)

if __name__ == "__main__":
    print("hello world")
    data = pickle.load(open('data_proc/wtv_names_10/X_10_wtv_stories_replaced.pkl','rb'))
    # data = data[0:-10]
    Y = pickle.load(open('data_proc/wtv_names_10/order_10_wtv_stories_replaced.pkl','rb'))
    # print(data[1])
    # print(Y[1])
    # exit()
    data, X_test, Y, y_test = train_test_split(data, Y, test_size=0.2)
    # x = data[1].reshape(1, 10, 300)
    # y = Y[1].reshape(1, 10)
    # print(y, x)
    # for i in range(10):
    #     x, y, yi = createfit(x, y, i)
    #     a = input()
    #     print(y, x)
    ypred = np.zeros((y_test.shape[0], y_test.shape[1]))

    models = []
    for i in range(max_sentence):
        models.append(build_model())
    for i in range(max_sentence):
        print(i,"i********************************************i", i)
        data, Y, Yi = createfit(data, Y, i)
        X_test, y_test, y_testi = createfit(X_test, y_test, i)
        # if (i==0):
        #     continue
        #     epochs = 25
        models[i].fit(data, Yi,
            batch_size=64,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_testi))
        predp = models[i].predict(X_test, verbose=0)
        for k in range(y_test.shape[0]):
            req = np.argmax(predp[k])
            ypred[k][i] = y_test[k][req]
        print(i,"i********************************************i", i)
        # print(ypred[0])
    
    f = open('predictions/predicted_e10_wtv_stories.pkl', 'wb')
    pickle.dump(ypred, f, protocol=2)
    f = open('predictions/real_e10_wtv_stories.pkl', 'wb')
    pickle.dump(y_test, f, protocol=2)
    