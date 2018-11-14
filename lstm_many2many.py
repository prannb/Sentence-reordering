from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import random as ran
import pickle

from keras.models import Sequential
from keras.layers import LSTM,Dense, Activation, Lambda, TimeDistributed
from sklearn.model_selection import train_test_split

# from list2datset import format_data
#--------------------Model Variables-----------
sen_vec_len = 300
max_sent =10
NUM_CLASS = max_sent
n_epochs = 30



def apply_model(model, X_train, Y_train, X_valid, Y_valid):
  pkl_filename = "unscramble.pkl" 
  print(model.summary())
  model.fit(X_train, Y_train, batch_size =1, epochs = n_epochs,  verbose = 5)
  # always write here, best models handled by hand to avoid over-writing
  with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  with open(pkl_filename, 'rb') as file:  
    pickle.load(file)
  X_test = X_valid[0:2,:,:]
  Y_test = Y_valid[0:2,:,:]
  for i in range(X_test.shape[0]):
    yhat = np.squeeze(model.predict(X_valid[i:i+1,:,:]))
    Y_pred = np.zeros_like(yhat)
    Y_pred[np.arange(len(yhat)),yhat.argmax(1)]=1
    print(Y_pred[0,0:30])
    # print(Y_test[i,0:7,0:7])

def lstm_many2many():
  # X,Y,order = format_data()
  # with open('data/X_10_wtv_stories_replaced.pkl','wb') as file:
  #   pickle.dump(X, file)
  # with open('data/Y_10_wtv_stories_replaced.pkl','wb') as file:
  #   input_lists = pickle.dump(Y, file) 
  train = True
  with open('data/X_10_wtv_stories_replaced.pkl','rb') as file:  
    X = pickle.load(file)
  with open('data/Y_10_wtv_stories_replaced.pkl','rb') as file:  
    Y = pickle.load(file)

  # X = X[0:200,:,:]
  # Y = Y[0:200,:,:]
  X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.1)
  print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
  if train:
    model = Sequential()
    model.add(LSTM(max_sent, input_shape = (max_sent,sen_vec_len),  return_sequences=True))
    model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax')))               
    # model.add(Lambda(lambda x: x[:, -20:, :]))
    # model.add(Dense(20,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
    apply_model(model, X_train, Y_train, X_valid, Y_valid)
  else:
    print("Loading Model...")
    model = pickle.load(open('unscramble.pkl','rb'))
    yhat = np.squeeze(model.predict(X_train[1000:1001,:,:]))
    Y_pred = np.zeros_like(yhat)
    Y_pred[np.arange(len(yhat)),yhat.argmax(1)]=1
    print(Y_pred[0:7,0:7])
    print(Y_train[1000:1001,0:7,0:7])

def main():
	lstm_many2many()

if __name__ == '__main__':
    main()