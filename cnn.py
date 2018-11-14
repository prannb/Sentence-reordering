import numpy as np
import tensorflow as tf
import pickle
import pdb    
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

epochs = 10

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    input_shape=(30,300,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(900, activation='relu'))

    model.compile(loss=categorical_crossentropy,
                optimizer=Adam(),
                metrics=['accuracy'])
    return model

def create_out(y):
    n = y.shape[0]
    ny = np.ones((n,900))
    i = 0
    for each in y:
        for j in range(30):
            ny[i][j*30: j*30+30] = each[j]
        i += 1
    return ny


if __name__ == "__main__":
    print("hello world")
    data = pickle.load(open('X_30_recc.pkl','rb'))
    Y = pickle.load(open('Y_30_recc.pkl','rb'))
    data = data
    Y = Y
    data, X_test, Y, y_test = train_test_split(data, Y, test_size=0.2)
    y_test = create_out(y_test)
    Y = create_out(Y)
    data = data.reshape(data.shape[0], 30, 300, 1)
    X_test = X_test.reshape(X_test.shape[0], 30, 300, 1)
    print(data.shape)
    print(y_test.shape)

    model = build_model()
    model.fit(data, Y,
                batch_size=64,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
