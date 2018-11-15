from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pickle
from dataset2list import create_list_nips
from dataset2list import create_list

max_sent = 10

def clean_para(text):
    sentences = sent_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stop_words = stopwords.words('english')
    data = []
    l = len(sentences)
    if l > max_sent:
        l = max_sent
    for i in range(l):
        sentence = sentences[i]
        tokens = word_tokenize(sentence)
        words = [word for word in tokens if word.isalpha()]
        tokens = [w.lower() for w in tokens]
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        data.append(words)
    # print(data)
    return data

def get_dataset():
    # text_list = create_list(filename = 'data/python3_data_names.tsv')
    text_list = create_list_nips()
    # print(text_list)
    data = []   
    for ind, text in enumerate(text_list):
        print(ind)
        data.append(clean_para(text))
    return data

def main():
    get_dataset()


if __name__ == '__main__':
    print("Cleaning the data")
    main()
    
