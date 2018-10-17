from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import random as ran

input_lists = [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
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
  				['might', 'know', 'child', 'forbidde', 'hospital', 'bed', 'driveby'], 
  				['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
  				['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
  				['thank', 'listening']]


def list2mat(input_lists = input_lists, vec_size = 300):
  #returns sentence matrix numb_sent x vec_size, randomly shuffled matrix, shuffling order

  n_sent = len(input_lists)
  sen_mat = np.zeros((n_sent, vec_size))
  shuff_sen_mat = np.zeros((n_sent, vec_size))
  shuff_order = list(range(n_sent))
  ran.shuffle(shuff_order)
  documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_lists)]
  model = Doc2Vec(documents, vector_size=300, window=2, min_count=1, workers=4)
  for i,sent in enumerate(input_lists):
    sen_mat[i,:] = model.infer_vector(input_lists[0])
  print(sen_mat.shape)
  for i in range(n_sent):
    shuff_sen_mat[i,:] = sen_mat[shuff_order[i]]

  return sen_mat, shuff_sen_mat, shuff_order

list2mat()