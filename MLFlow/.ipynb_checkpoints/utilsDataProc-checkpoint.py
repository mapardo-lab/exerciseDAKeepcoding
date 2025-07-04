import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import unicodedata
from num2words import num2words
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import re
import pickle

def read_raw_data(file):
    data = pd.read_json(file, lines=True)
    return data

def split_data(data):
    reviewText_train, reviewText_test, overall_train, overall_test = train_test_split(
        data['reviewText'], data['overall'],
        train_size=0.75, test_size=0.25,
        random_state=42, shuffle=True
    )

    return reviewText_train, reviewText_test, overall_train, overall_test

def normalize_ASCII(text):
  """
  Normalizes Unicode text to its ASCII representation by decomposing complex characters,
  removing non-ASCII characters, and returning a clean UTF-8 string.
  """
  text = unicodedata.normalize('NFKD', text) # Break down Unicode Characters
  text = text.encode('ascii', 'ignore') # Convert to ASCII
  text = text.decode('utf-8', 'ignore') # Decode back to UTF-8
  return text

def cleanning(text):
  """
  Cleans text by:
  1. Removing all punctuation except apostrophes (for contractions)
  2. Eliminating single-letter words
  """
  text = re.sub(r"[^a-z0-9']", ' ', text) # Remove punctuation. Only words, spaces and ' are kept. Keep ' is important for removing stopwords step.
  text = re.sub(r" [a-z] ", ' ', text) # Remove one letter words
  return text

def remove_stopwords(text, stopwords):
  """
  Removes stopwords and apostrophes from input text, preserving meaningful words.
  """
  list_words = []
  for word in text.split():
    if word not in stopwords: # Remove stopwords
      list_words.append(word)
  cleaned_text = ' '.join(list_words)
  cleaned_text = re.sub(r"'", '', cleaned_text) # Remove punctuation '
  return cleaned_text

def numbers2words(text):
  """
  Converts all numeric digits in a text string to their word equivalents
  """
  list_words = []
  for word in text.split():
    if word.isdigit():
      list_words.append(num2words(word, ordinal=False))
    else:
      list_words.append(word)
  new_text = ' '.join(list_words)
  return new_text

def stemming(text):
  """
  Applies Porter stemming to each word in the input text, reducing words to their root forms.
  """
  stemmer = PorterStemmer()
  list_words = []
  for word in text.split():
    list_words.append(PorterStemmer().stem(word))
  return ' '.join(list_words)

def overall2label(overall):
  """
  Converts a numerical 'overall' rating into a binary label:
  - 0 for ratings below 4 (negative)
  - 1 for ratings 4 or above (positive)
  """
  label = None
  if overall < 4:
    label = 0
  else:
    label = 1
  return label


def review2words(text):
  """
  Applies a complete text preprocessing pipeline to normalize and clean input text.
  Performs the following transformations in sequence:
  1. Converts to lowercase
  2. Normalizes Unicode to ASCII
  3. Cleans punctuation and single-letter words
  4. Removes stopwords
  5. Applies stemming
  6. Converts numbers to words
  """
  text = text.lower() # To lowercase
  text = normalize_ASCII(text)
  text = cleanning(text)
  text = remove_stopwords(text, STOPWORDS)
  text = stemming(text)
  text = numbers2words(text)
  words = text.split()
  return words

def preprocess_data(data_train, data_test, labels_train, labels_test):
    """
    Preprocesses training and test data by:
    1. Converting reviews to tokenized words
    2. Transforming ratings to binary labels
    """
    words_train = list(map(review2words, data_train))
    words_test = list(map(review2words, data_test))
    labels_train = list(map(overall2label, labels_train))
    labels_test = list(map(overall2label, labels_test))

    file = 'cache/proc_data.pkl'
    cache_data = dict(words_train=words_train, words_test=words_test,
                      labels_train=labels_train, labels_test=labels_test)
    with open(file, "wb") as f:
              pickle.dump(cache_data, f)
    print(f"Wrote preprocessed data to cache file: {file}")

    return words_train, words_test, labels_train, labels_test

def read_proc_data(file):
    try:
        with open(file, "rb") as f:
                cache_data = pickle.load(f)
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
        print("Read preprocessed data from cache file:", file)
    except:
        pass
    return  words_train, words_test, labels_train, labels_test 
    