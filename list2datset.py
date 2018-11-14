import numpy as np
import random as ran
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from cleaning_whole import get_dataset
#--------------------Model Variables-----------
sen_vec_len = 300
max_sent = 5
NUM_CLASS = max_sent

# embedding = 'doc2vec'
embedding = 'word2vec'

# input_lists = get_dataset()
# pkl_filename = 'input_lists.pkl'
# pkl_filename = 'input_lists_names_10.pkl'
pkl_filename = 'input_lists_names_5.pkl'

# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(input_lists, file)
with open(pkl_filename, 'rb') as file:  
    input_lists = pickle.load(file)
# input_lists = [[['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
#  				['thing'], 
#  				['dont', 'think'],
#   				['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
#   				['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
#   				['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
#   				['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
#   				['learn', 'countrysstates', 'outside'], 
#   				['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
#   				['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
#   				['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
#   				['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
#   				['might', 'know', 'child', 'forbidde', 'hospital', 'bed', 'driveby'], 
#   				['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
#   				['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
#   				['thank', 'listening']],
#           [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
#         ['thing'], 
#         ['dont', 'think'],
#           ['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
#           ['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
#           ['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
#           ['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
#           ['learn', 'countrysstates', 'outside'], 
#           ['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
#           ['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
#           ['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
#           ['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
#           ['might', 'know', 'child', 'forbidde', 'hospital', 'bed', 'driveby'], 
#           ['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
#           ['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
#           ['thank', 'listening']],
#           [['dear', 'local', 'newspaper', 'think', 'effects', 'computers', 'people', 'great', 'learning', 'skillsaffects', 'give', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'astronomy', 'keeps', 'us', 'troble'],
#         ['thing'], 
#         ['dont', 'think'],
#           ['would', 'feel', 'teenager', 'always', 'phone', 'friends'], 
#           ['ever', 'time', 'chat', 'friends', 'buisness', 'partner', 'things'], 
#           ['well', 'new', 'way', 'chat', 'computer', 'plenty', 'sites', 'internet', 'facebook', 'myspace', 'ect'], 
#           ['think', 'setting', 'meeting', 'boss', 'computer', 'teenager', 'fun', 'phone', 'rushing', 'get', 'cause', 'want', 'use'],
#           ['learn', 'countrysstates', 'outside'], 
#           ['well', 'computerinternet', 'new', 'way', 'learn', 'going', 'time'], 
#           ['might', 'think', 'child', 'spends', 'lot', 'time', 'computer', 'ask', 'question', 'economy', 'sea', 'floor', 'spreading', 'even', 'surprise', 'much', 'heshe', 'knows'], 
#           ['believe', 'computer', 'much', 'interesting', 'class', 'day', 'reading', 'books'], 
#           ['child', 'home', 'computer', 'local', 'library', 'better', 'friends', 'fresh', 'perpressured', 'something', 'know', 'isnt', 'right'], 
#           ['might', 'know', 'child', 'forbidde', 'hospital', 'bed', 'driveby'], 
#           ['rather', 'child', 'computer', 'learning', 'chatting', 'playing', 'games', 'safe', 'sound', 'home', 'community', 'place'], 
#           ['hope', 'reached', 'point', 'understand', 'agree', 'computers', 'great', 'effects', 'child', 'gives', 'us', 'time', 'chat', 'friendsnew', 'people', 'helps', 'us', 'learn', 'globe', 'believe', 'keeps', 'us', 'troble'], 
#           ['thank', 'listening']]]
if embedding == 'word2vec':
  print("Loading Word2vec Model....")
  wmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
  print("Loaded")

def infer_avg_word_vec(sen_words, vec_size = 300):
  ndoc = len(sen_words)
  wtv = np.zeros([300])
  co = 0
  for w in sen_words:
    if w in wmodel.vocab:
      wtv = wtv + wmodel.wv[w]
      co = co + 1
  if co ==0:
    co =1
  wtv = wtv/co
  return wtv

def list2mat_word_vec(input_lists = input_lists, vec_size = 300):
  #returns sentence matrix numb_sent x vec_size, randomly shuffled matrix, shuffling order

  n_sent = len(input_lists)
  sen_mat = np.zeros((n_sent, vec_size))
  shuff_sen_mat = np.zeros((n_sent, vec_size))
  shuff_order = list(range(n_sent))
  ran.shuffle(shuff_order)
  for i,sent in enumerate(input_lists):
      sen_mat[i,:] = infer_avg_word_vec(input_lists[i])
  # print(sen_mat.shape)
  for i in range(n_sent):
    shuff_sen_mat[i,:] = sen_mat[shuff_order[i]]
  return sen_mat, shuff_sen_mat, shuff_order

def list2mat_doc_vec(doc2vec_model, input_lists = input_lists, vec_size = 300):
  #returns sentence matrix numb_sent x vec_size, randomly shuffled matrix, shuffling order
  n_sent = len(input_lists)
  sen_mat = np.zeros((n_sent, vec_size))
  shuff_sen_mat = np.zeros((n_sent, vec_size))
  shuff_order = list(range(n_sent))
  ran.shuffle(shuff_order)
  for i,sent in enumerate(input_lists):
    sen_mat[i,:] = doc2vec_model.infer_vector(input_lists[i])
  # print(sen_mat.shape)
  for i in range(n_sent):
    shuff_sen_mat[i,:] = sen_mat[shuff_order[i]]
  return sen_mat, shuff_sen_mat, shuff_order

def order2output_lstm(c):
  op = np.identity(max_sent)
  order = c
  for ind,ele in enumerate(order):
    d = order[0:ind+1]
    d.sort()
    op[ind,ind] = 0
    op[ind,d.index(ele)] = 1 
  return op

def order2output_prann(c):
  op = np.identity(max_sent)
  order = c
  for ind,ele in enumerate(order):
    op[ind,ind] = 0
    op[ind,order.index(ind)] = 1 
  return op

def format_data(input_lists = input_lists):
  n = len(input_lists)
  n_sen_mat = np.zeros((n,max_sent,sen_vec_len))
  n_shuff_sen_mat = np.zeros((n,max_sent,sen_vec_len))
  Y = np.zeros((n,max_sent,max_sent))
  order = np.zeros((n,max_sent))
  order = order-1
  if embedding == 'doc2vec':
    sentences = []
    # for point in input_lists:
    #   for sent in point:
    #     sentences.append(sent)
    # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    # doc2vec_model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)  
    with open('models/doc2vec_model.pkl','rb') as file:
      doc2vec_model = pickle.load(file)
  index = 0
  for i in range(len(input_lists)):
    print(i)
    if embedding == 'doc2vec':
      a,b,c = list2mat_doc_vec(doc2vec_model, input_lists[i])
    elif embedding == 'word2vec':
      a,b,c = list2mat_word_vec(input_lists[i])
    n_sent_point = len(c)
    if n_sent_point >= max_sent:
      order[index,0:len(c)] = c
      n_sen_mat[index,0:a.shape[0],0:a.shape[1]] = a
      n_shuff_sen_mat[index,0:b.shape[0],0:b.shape[1]] = b
      Y[index,:,:] = order2output_lstm(c)
      # Y[i,:,:] = order2output_prann(c)
      # Y[k,x,y] shows that kth paragraph's xth shuffled sentence is at yth position in original sentence
      index = index + 1
  print(index)
  n_shuff_sen_mat = n_shuff_sen_mat[0:index,:,:]
  Y = Y[0:index,:,:]
  order = order[0:index,:]
  return n_shuff_sen_mat,Y,order
