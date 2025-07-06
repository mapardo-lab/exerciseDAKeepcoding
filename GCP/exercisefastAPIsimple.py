from fastapi import FastAPI
from class_values import VehicleData
import pandas as pd
from pysentimiento import create_analyzer
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sklearn.datasets import fetch_california_housing

app = FastAPI()

# data for superhero movies
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing['data'], columns = housing['feature_names'])

@app.get('/california_housing')
def search_housing(population: float, occup: float):
    total_district = df_housing.shape[0]
    num_district = df_housing[(df_housing['Population'] > population) & (df_housing['AveOccup'] > occup)].shape[0]
    prop = num_district/total_district
    return {'message': prop}

