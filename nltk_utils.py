import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
     dividir la frase en un conjunto de palabras/tokens
    un token puede ser una palabra o un carácter de puntuación, o un número
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = encontrar la forma de la raíz de la palabra
    ejemplos:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    devuelve la matriz de palabras:
    1 para cada palabra conocida que exista en la frase, 0 en caso contrario
    ejemplo:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # acortar cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializar la bolsa con 0 para cada palabra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag