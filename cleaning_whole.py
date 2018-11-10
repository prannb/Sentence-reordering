from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from dataset2list import create_list


def clean_para(text):
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

def get_dataset():
    text_list = create_list()
    data = []
    for text in text_list:
        data.append(main(text))
    return data d

def main():
    get_dataset()


if __name__ == '__main__':
    print("Cleaning the data")
<<<<<<< HEAD:cleaning_whole.py
    main()
    
=======
    # file = open(filename, 'r')
    # text = file.read()
    # file.close()
    text = "some text"
    data = main(text)
    print(data)
>>>>>>> 2cab9d048362c2f7d680ef73fbfc7c573ed32db7:cleaning.py
