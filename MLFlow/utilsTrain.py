import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import mlflow
from sklearn.model_selection import StratifiedKFold
from statistics import mean

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