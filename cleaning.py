from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


def main(text):
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




if __name__ == '__main__':
    print("Cleaning the data")
    # file = open(filename, 'r')
    # text = file.read()
    # file.close()
    text = "some text"
    data = main(text)
    print(data)