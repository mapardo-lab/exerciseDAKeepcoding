import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

def extract_BoW_features(words_train, words_test):
    """
    Extract Bag-of-Words for a given set of documents
    Input is already preprocessed into words.
    """
    vectorizer = CountVectorizer(ngram_range=(1,2),
                                 min_df = 0.001,
                                 preprocessor=lambda x: x, tokenizer=lambda x: x)  # already preprocessed
    vectorizer.fit(words_train)
    features_train = vectorizer.transform(words_train)
    features_test = vectorizer.transform(words_test)
    vocabulary = vectorizer.vocabulary_

    return features_train, features_test, vocabulary

#def features_selection():
#    # prepare X_data (1. feature selection)
#    features_index = [index for feature, index in vocabulary.items() if feature in features_sel]
#    features_select_train = features_train.toarray()[:,features_index]
#    features_select_test = features_test.toarray()[:,features_index]

def prepare_data(features_train, features_test, labels_train, labels_test):
    # prepare X_data (2. normalize bag-of-words by row)
    X_train = normalize(features_train, axis=1)
    X_test = normalize(features_test, axis=1)

    # prepare y_data
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test

