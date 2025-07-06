## exerciseDAKeepcoding

### MLFlow

Se toma como punto de partida la práctica realizada para el módulo de NLP. En esta práctica se entrenó un modelo para el reconocimiento de sentimiento a partir de un conjunto de 'reviews' clasificadas en positivas y negativas.

Los resultados obtenidos se adaptan a esta práctica con el objetivo de buscar el mejor modelo de KNN para la clasificación de sentimiento de las 'reviews'.

El procedimiento seguido en la práctica incluye primero la exploración de los diferentes pasos mediante notebooks y una vez definido el protocolo a seguir, se genera un script main.py junto con sus librerías.

- 1_EDA.ipynb, análisis exploratorio de datos

- 2_Data_Proc.ipynb, procesado de datos

- 3_MLA.ipynb, búsqueda del mejor modelo KNN

- main.py, script para la búsqueda del mejor modelo KNN. Imprescindible un argumento de entrada que puede ser el fichero de datos original o un fichero cache con el procesado de los datos

- utilsDataProc.py, utilsFeatures.py, utilsTrain.py, librerías de funciones usadas en main.py

Se adjuntan las capturas de pantalla en MLFlow_images.pdf.

### fastAPI

Se consruye una API con la librería fastAPI de python que contiene 5 'apps' (exercisefastAPI.py):

- vehicle-info, app.post para introducir los datos de un vehículo (matrícula, modelo , color). Se difine una clase VehicleData que contiene la definición de tipo de las variables (class_values.py)

- california-housing, consulta al conjunto de datos 'california housing' de scikit-learn. La aplicación solicita dos valores y a partir de ellos se cálcula el número de distritos que tienen una población y una proporción de ocupación por encima de esos valores.

- pokemon-search, app.get que a partir del tipo de pokemon (Water, Fire, ...) y número total de puntos, te devuelve los nombres de los pokemons que de ese tipo y que tienen un número de puntos mayor que el valor dado.

- get-sentiment, app.get que utiliza el modelo 'pysentimiento/robertuito-sentiment-analysis' descargado de HuggingFace para analizar el setimiento de texto (POSITIVO, NEUTRO O NEGATIVO).

- translate, app.get que utiliza el modelo 'facebook/mbart-large-50-many-to-many-mmt' descargado de HuggingFace para traducir un texto del español al inglés.

Se adjuntan las capturas de pantalla de la aplicación en fastaAPIimages.pdf.
