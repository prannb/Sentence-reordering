from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import random as ran
import pickle

from keras.models import Sequential
from keras.layers import LSTM,Dense, Activation, Lambda, TimeDistributed
from sklearn.model_selection import train_test_split

from cleaning import get_dataset
#--------------------Model Variables-----------
sen_vec_len = 300
max_sent = 20
NUM_CLASS = 20

# input_lists = get_dataset()
input_lists = [[['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
 				['thing'], 
 				['dont', 'think'],
  				['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
  				['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
  				['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
  				['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
  				['learn', 'countrysstates', 'outside'], 
  				['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
  				['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
  				['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
  				['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
  				['might', 'know', 'child', 'forbidden', 'hospital', 'bed', 'driveby'], 
  				['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
  				['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
  				['thank', 'listening']],
          [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
          ['thing'], 
          ['dont', 'think'],
          ['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
          ['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
          ['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
          ['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
          ['learn', 'countrysstates', 'outside'], 
          ['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
          ['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
          ['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
          ['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
          ['might', 'know', 'child', 'forbidden', 'hospital', 'bed', 'driveby'], 
          ['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
          ['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
          ['thank', 'listening']],
          [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
          ['thing'], 
          ['dont', 'think'],
          ['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
          ['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
          ['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
          ['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
          ['learn', 'countrysstates', 'outside'], 
          ['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
          ['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
          ['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
          ['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
          ['might', 'know', 'child', 'forbidden', 'hospital', 'bed', 'driveby'], 
          ['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
          ['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
          ['thank', 'listening']]]


def list2mat(input_lists = input_lists, vec_size = sen_vec_len):
  #returns sentence matrix numb_sent x vec_size, randomly shuffled matrix, shuffling order

  n_sent = len(input_lists)
  sen_mat = np.zeros((n_sent, vec_size))
  shuff_sen_mat = np.zeros((n_sent, vec_size))
  shuff_order = list(range(n_sent))
  ran.shuffle(shuff_order)
  documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_lists)]
  model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)
  for i,sent in enumerate(input_lists):
    sen_mat[i,:] = model.infer_vector(input_lists[i])
  for i in range(n_sent):
    shuff_sen_mat[i,:] = sen_mat[shuff_order[i]]

  return sen_mat, shuff_sen_mat, shuff_order

def order2output(c):
  op = np.identity(max_sent)
  order = c
  for ind,ele in enumerate(order):
    d = order[0:ind+1]
    d.sort()
    op[ind,ind] = 0
    op[ind,d.index(ele)] = 1 
  return op

def format_data(input_lists = input_lists):
  n = len(input_lists)
  n_sen_mat = np.zeros((n,max_sent,sen_vec_len))
  n_shuff_sen_mat = np.zeros((n,max_sent,sen_vec_len))
  Y = np.zeros((n,max_sent,max_sent))
  for i in range(len(input_lists)):
    a,b,c = list2mat(input_lists[i])
    n_sen_mat[i,0:a.shape[0],0:a.shape[1]] = a
    n_shuff_sen_mat[i,0:b.shape[0],0:b.shape[1]] = b
    Y[i,:,:] = order2output(c)# Y[k,x,y] shows that kth paragraph's xth shuffled sentence is at yth position in original sentence
  return n_shuff_sen_mat,Y