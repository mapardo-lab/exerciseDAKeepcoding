import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, make_scorer, classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import joblib
import mlflow
from sklearn.model_selection import StratifiedKFold
from statistics import mean


def read_raw_data(file):
    data = pd.read_json(file, lines=True)
    return data

def split_data(data)
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



def mlflow_knn_n(name_experiment, X_train, y_train, list_n):
    mlflow.set_experiment(f'{name_experiment}_knn')
    kfolds  = StratifiedKFold(n_splits = 5, shuffle = True, random_state=42)
    for n in list_n:
        model = KNeighborsClassifier(n_neighbors=n)
        score_val, score_train = modelfitCV(model, kfolds, X_train, y_train)
        run_name = f'{name_experiment}-n_neighbors_{n}'
        print(f"Running {run_name}")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric('Score Train', score_train)
            mlflow.log_metric('Score Validation', score_val)
            mlflow.log_param('Num neighbors', n)

def modelfitCV(model, kfolds, X_train, y_train):
    score_train = []
    score_val = []
    for idxTrain, idxVal in kfolds.split(X_train,y_train): 
        Xt = X_train[idxTrain,:]
        yt = y_train[idxTrain]
        Xv = X_train[idxVal,:]
        yv = y_train[idxVal]
        model.fit(Xt,yt)
        score_val.append(model.score(Xv, yv))
        score_train.append(model.score(Xt, yt))
    mean_score_val = mean(score_val)
    mean_score_train = mean(score_train)

    return mean_score_val, mean_score_train 

def read_data(cache_dir = "cache", cache_file = "train_model_data.pkl"):
    print(os.path.join(cache_dir, cache_file))
    try:
      with open(os.path.join(cache_dir, cache_file), "rb") as f:
            cache_data = joblib.load(f)
      print("Read preprocessed data from cache file:", cache_file)
    except:
      pass

    X_train = cache_data['X_train']
    X_test = cache_data['X_test']
    y_train = cache_data['y_train']
    y_test = cache_data['y_test']

    return X_train, y_train, X_test, y_test