from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import random as ran
import pickle

from keras.models import Sequential
from keras.layers import LSTM,Dense, Activation, Lambda, TimeDistributed
from sklearn.model_selection import train_test_split

from list2datset import format_data
#--------------------Model Variables-----------
sen_vec_len = 300
max_sent = 30
NUM_CLASS = 30
n_epochs = 30

def apply_model(model, X_train, Y_train, X_valid, Y_valid):
  print(model.summary())
  model.fit(X_train, Y_train, batch_size =1, epochs = n_epochs,  verbose = 5)

  pkl_filename = "unscramble.pkl" # always write here, best models handled by hand to avoid over-writing
  with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)

  yhat = np.squeeze(model.predict(X_valid))
  print(yhat)
  Y_pred = np.zeros_like(yhat)
  Y_pred[np.arange(len(yhat)),yhat.argmax(1)]=1

  print(Y_pred)
  print(Y_train)

def lstm_many2many():
  # X,Y = format_data()
  # with open('data/X.pkl','wb') as file:
  #   pickle.dump(X, file)
  # with open('data/Y.pkl','wb') as file:
  #   input_lists = pickle.dump(Y, file) 
  with open('data/X_30.pkl','rb') as file:  
    X = pickle.load(file)
  with open('data/Y_30.pkl','rb') as file:  
    Y = pickle.load(file) 
  X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.1)
  print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
  model = Sequential()
  model.add(LSTM(max_sent, input_shape = (max_sent,sen_vec_len),  return_sequences=True))
  model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax')))               
  # model.add(Lambda(lambda x: x[:, -20:, :]))
  # model.add(Dense(20,activation='softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
  apply_model(model, X_train, Y_train, X_valid, Y_valid)

def main():
	lstm_many2many()

if __name__ == '__main__':
    main()