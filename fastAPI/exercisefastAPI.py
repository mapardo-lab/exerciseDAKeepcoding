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

# data for pokemons
file_path = "Pokemon.csv"
df_pokemons = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                            "abcsds/pokemon",file_path)

# model for sentiment analysis
analyzer = create_analyzer(task="sentiment", lang="es")

# translation
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

@app.post('/vehicle-info')
def register_user(vehicle: VehicleData):
    return {'message': f'{vehicle.number_plate}: {vehicle.model}'}

@app.get('/california_housing')
def search_housing(population: float, occup: float):
    total_district = df_housing.shape[0]
    num_district = df_housing[(df_housing['Population'] > population) & (df_housing['AveOccup'] > occup)].shape[0]
    prop = num_district/total_district
    return {'message': prop}

@app.get('/pokemon_search')
def search_pokemon(type_pokemon: str, total: int):
    pokemons = df_pokemons[(df_pokemons['Type 1'] == type_pokemon) & (df_pokemons['Total'] > total)]['Name']
    return {'message': pokemons}
                
@app.get('/get-sentiment')
def sentiment(text: str):
    result = analyzer.predict(text)

    return {'sentiment': result}
    
@app.get('/translate')
def translator(text: str):
    tokenizer.src_lang = "es_XX"
    encoded_hi = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return {'translation': result}

