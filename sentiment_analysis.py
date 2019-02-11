# Authors: Alexandre Gramfort
#          Chloe Clavel
# License: BSD Style.
# TP Cours ML Telecom ParisTech MDI343

import os.path as op
import numpy as np
import string

from sklearn.base import BaseEstimator, ClassifierMixin

###############################################################################
# Load data
print("Loading dataset")

from glob import glob
filenames_neg = sorted(glob(op.join('data', 'imdb1', 'neg', '*.txt')))
filenames_pos = sorted(glob(op.join('data', 'imdb1', 'pos', '*.txt')))

texts_neg = [open(f).read() for f in filenames_neg]
texts_pos = [open(f).read() for f in filenames_pos]
texts = texts_neg + texts_pos
y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))

###############################################################################
# Start part to fill in

def count_words(texts):
    """Vectorize text : return count of each word in the text snippets

    Parameters
    ----------
    texts : list of str
        The texts

    Returns
    -------
    vocabulary : dict
        A dictionary that points to an index in counts for each word.
    counts : ndarray, shape (n_samples, n_features)
        The counts of each word in each text.
        n_samples == number of documents.
        n_features == number of words in vocabulary.
    """
    words = set()
    vocabulary = {}
    table = str.maketrans({key:" " for key in string.punctuation})
    i = 0
    j = 0

    for text in texts:
        word_list = text.translate(table).lower().split(" ")
        for word in word_list:
            if word not in words:
                words.add(word)
                vocabulary[word] = j
                j += 1

    n_features = len(words)
    counts = np.zeros((len(texts), n_features))

    for text in texts:
        word_list = text.translate(table).lower().split(" ")
        for word in word_list:
            counts[i][vocabulary[word]] += 1
        i += 1

    return vocabulary, counts


class NB(BaseEstimator, ClassifierMixin):
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.prior = np.zeros((2))
        self.condprob = None
        self.scores = None

    def fit(self, X, y):
        self.condprob = np.zeros((2, X.shape[1]))
        for c in [0,1]:
            self.prior[c] = X[y==c].shape[0] / X.shape[0]
            self.condprob[c,:] = (np.sum(X[y==c], axis=0) +1) / np.sum(np.sum(X[y==c], axis=1)+1)
        return self.vocabulary, self.prior, self.condprob


    def predict(self, X):
        self.scores = np.zeros((X.shape[0], self.prior.shape[0]))
        self.scores += np.log(self.prior)
        tmp = np.zeros((X.shape[0], X.shape[1], 2))
        for c in [0,1]:
            tmp[:,:,c] = np.multiply(X, self.condprob[c,:])
        tmp[tmp==0] = 1
        self.scores += np.sum(np.log(tmp), axis=1)
        return np.argmax(self.scores, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Count words in text
vocabulary, X = count_words(texts)

# Try to fit, predict and score
nb = NB(vocabulary)
nb.fit(X[::2], y[::2])
print (nb.score(X[1::2], y[1::2]))
