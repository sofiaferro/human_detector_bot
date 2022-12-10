import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

# instance stemmer
stemmer = PorterStemmer()

# tokenization
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# stemming
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tolkenized_sentence, all_words):
    tokenize_sentence = [stem(w) for w in tolkenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for ind, w, in enumerate(all_words):
        if w in tokenize_sentence:
            bag[ind] = 1.0
    return bag
