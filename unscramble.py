import pickle
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import random as ran
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


sen_vec_len = 300
max_sent = 30

def tokenise_raw(text):
    sentences = sent_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stop_words = stopwords.words('english')
    data = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [word for word in tokens if word.isalpha()]
        tokens = [w.lower() for w in tokens]
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        data.append(words)
    # print(data)
    return data

# print("Loading....")
# wmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
# print("loaded")

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

def list2mat(input_list, vec_size = sen_vec_len):
  #returns sentence matrix numb_sent x vec_size, randomly shuffled matrix, shuffling order
  print(input_list)
  n_sent = len(input_list)
  # print(n_sent)
  sen_mat = np.zeros((n_sent, vec_size))
  shuff_sen_mat = np.zeros((n_sent, vec_size))
  shuff_order = list(range(n_sent))
  ran.shuffle(shuff_order)
  # print(shuff_order)
  # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(input_list)]
  # model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=1, workers=4)
  # for i,sent in enumerate(input_list):
  #   sen_mat[i,:] = model.infer_vector(input_list[i])
  for i,sent in enumerate(input_list):
    sen_mat[i,:] = infer_avg_word_vec(input_list[i])
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

def format_data(input_lists):
	n = len(input_lists)
	n_sen_mat = np.zeros((n,max_sent,sen_vec_len))
	n_shuff_sen_mat = np.zeros((n,max_sent,sen_vec_len))
	Y = np.zeros((n,max_sent,max_sent))
	# Y = np.zeros((n,20))
	for i in range(len(input_lists)):
		# print(i,len(input_lists[0]))
		a,b,c = list2mat(input_lists[i])
		n_sen_mat[i,0:a.shape[0],0:a.shape[1]] = a
		n_shuff_sen_mat[i,0:b.shape[0],0:b.shape[1]] = b
		# Y[i,:] = range(max_sent)
		Y[i,:,:] = order2output(c)# Y[k,x,y] shows that kth paragraph's xth shuffled sentence is at yth position in original sentence
		# for ind,pos in enumerate(c):
		#   Y[i,ind,pos] = 1
		# Y[i,c.index(0)] = 1
	return n_shuff_sen_mat,Y

def load_model_nd_apply(X,Y):
	pkl_filename = 'unscramble.pkl'
	with open(pkl_filename, 'rb') as file:  
		model = pickle.load(file)
	yhat = np.squeeze(model.predict(X))
	Y_pred = np.zeros_like(yhat)
	Y_pred[np.arange(len(yhat)),yhat.argmax(1)]=1
	print(Y_pred[0:7,0:7])
	print(Y[0,0:7,0:7])



def main():
  # text = ["He was too busy. Arjun asked permission to leave. They left after we allowed him to do so. He is a fool"]
  # # print(tokenise_raw(text[0]))
  # input_lists = [tokenise_raw(text[0])]
  # print(input_lists)
  # print(len(input_lists))
  # print(len(input_lists[0]))
  # X, Y = format_data(input_lists)
  with open('data/X_30_lstm.pkl','rb') as file:  
    X = pickle.load(file)
  with open('data/Y_30_lstm.pkl','rb') as file:  
    Y = pickle.load(file)
  X = X[3:4,:,:]
  Y = Y[3:4,:,:]
  load_model_nd_apply(X,Y)

if __name__ == '__main__':
  main()