from fastapi import FastAPI
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# cargar el archivo CSV en un DataFrame
df = pd.read_csv('dataset_plataformas.csv',usecols=['id', 'type', 'title', 'cast', 'country','release_year', 'rating','listed_in', 'duration_int', 'duration_type', 'score'])
df_movie = pd.read_csv('df_ml.csv')
app = FastAPI()

#http://127.0.0.1:8000

##################################################################################################################################
@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    """
    Devuelve sólo el string del nombre de la película con mayor duración según año, plataforma y tipo de duración
    """
    #Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = platform.lower()[0]
    duration_type = duration_type.lower()
    resultado = df[(df['release_year']==year) & (df['id'].str.startswith(platform)) & (df['duration_type']== duration_type)]
    idx = resultado['duration_int'].idxmax()
    return {'pelicula': resultado.loc[idx, 'title']}


##################################################################################################################################
@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, year: int):
    ''' 
    Devuelve un int con el total de películas con puntaje mayor a XX en determinado año
    '''
    #Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = plataforma.lower()[0]
    #Filtro por año de lanzamiento, plataforma aclarando que solo tenga en cuenta la primer letra de la columna id y que el tipo sea 'movie'.
    data_filtrada = df[(df['id'].str.startswith(platform)) & (df['release_year'] == year) & (df['type'] == 'movie')]
    #Cuento la cantidad de películas que tienen el rating deseado
    cantidad = data_filtrada[data_filtrada['score'] > scored]['id'].count()
    return {'plataforma': plataforma,
            'cantidad': int(cantidad),
            'anio': year,
            'score': scored}
##################################################################################################################################
@app.get('/get_count_platform/{plataforma}')
def get_count_platform(platform: str):
    '''
    Devuelve un int con el número total de películas de la plataforma que ingresa
    '''
    #Me aseguro que si el usuario escribe en mayuscula se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = platform.lower()[0]
    #Realizo el filtro para que cuente del dataframe solo la plataforma de la columna ID y movie de la columna type
    cantidad = df[(df['id'].str.startswith(platform)) & (df['type'] == 'movie')].shape[0]
    return {'plataforma': platform, 'peliculas': cantidad}

##################################################################################################################################
@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(platforma: str, year: int):
    ''' 
    Devuelve sólo el  string con el nombre del actor que más se repite según la plataforma y el año dado
    '''
    #Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = platforma.lower()[0]
    df_filt = df[(df['id'].str.startswith(platform)) & (df['release_year'] == year)]
    #Utilizo la función dropna() para eliminar los valores nulos de la columna "cast" del DataFrame. 
    #Con el método apply() aplico la función str() a cada valor de la columna "cast", convirtiendo los valores a cadenas. 
    #Uso el método str.split(',') para dividir cada cadena en una lista de actores utilizando la coma como separador. 
    #Obtengo una serie de pandas que contiene una lista de actores para cada fila del DataFrame original.
    actores_por_fila = df_filt['cast'].dropna().apply(lambda x: [i.strip() for i in x.split(',') if not i.strip().isdigit()])
    #Cuento la cantidad de veces que aparece cada actor en todas las filas, utilizando la clase Counter de Python.
    contador_actores = Counter()
    for actores in actores_por_fila:
        contador_actores.update(actores)
    #Encuentro el actor que aparece más veces utilizando la funcion most_common devolviendo una lista de tuplas donde cada tupla contiene un actor 
    #y la cantidad de veces que aparece en todas las filas del DataFrame.
    actor_mas_repetido = contador_actores.most_common(1)
    if actor_mas_repetido:
        #Asigno [0][0] para indicar el actor que mas veces aparece
        actor_mas_repetido = actor_mas_repetido[0][0]
        #Muestro el actor que aparece más veces y la cantidad de veces que aparece
        cantidad_actor_mas_repetido = contador_actores[actor_mas_repetido]
        return {'plataforma': platforma,
                'anio': year,
                'actor': actor_mas_repetido,
                'apariciones': int(cantidad_actor_mas_repetido)}
    else:
        return {'plataforma': platforma,
                'anio': year,
                'actor': "No hay datos disponibles",
                'apariciones': "No hay datos disponibles"}

##################################################################################################################################
@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    ''' 
    Devuelve el numero total de contenido con ese rating de audiencias
    '''
    rating = rating.lower()
    #Filtro el dataframe por tipo, país y año
    df_filtrado = df[df['rating'] == rating]
    #Me quedo con la cantidad de producciones
    cantidad = len(df_filtrado)
    return {'rating': rating, 'contenido': cantidad}

##################################################################################################################################
@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_country(tipo: str, pais: str, year: int):
    ''' 
    Devuelve el tipo de contenido (pelicula,serie,documental) por pais y año en un diccionario con las variables 
    llamadas 'pais' (nombre del pais), 'anio' (año), 'pelicula' (tipo de contenido)
    '''
    #Filtro el dataframe por tipo, país y año
    df_filtrado = df[(df['type'] == tipo) & (df['country'] == pais) & (df['release_year'] == year)]
    #Me quedo con la cantidad de producciones
    cantidad = len(df_filtrado)
    return {'pais': pais,
            'anio': year,
            'contenido': int(cantidad)
            }

##################################################################################################################################

@app.get('/get_recomendation/{title}')
def get_recommendationB(title):
    ''' 
    Devuelve una lista con los títulos de las 5 películas más recomendadas según el título de película referencia
    el modelo de recomendación se basa en las similitudes de la categoría, valores de la columna 'listed_in'
    '''
    #Me aseguro de trabajar con la variable en minúscula
    title = title.lower()
    #Creo la matriz de similitud
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movie['listed_in'])
    #Busco el índice de la película referencia
    idx = df_movie.index[df_movie['title'] == title.lower()].tolist()[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    #Creo la lista de los títulos recomendados
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if i[0] != idx]
    #Ordeno la lista de mayor a menor según su 'score'
    sim_scores = sorted(sim_scores, key=lambda x: df_movie['score'].iloc[x[0]], reverse=True)[:5]
    respuesta = df_movie.iloc[[i[0] for i in sim_scores]]['title'].tolist()
    return {'recomendacion': respuesta}


##################################################################################################################################