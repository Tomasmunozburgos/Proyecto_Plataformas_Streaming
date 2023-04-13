from fastapi import FastAPI
import pandas as pd
import numpy as np
from collections import Counter

# cargar el archivo CSV en un DataFrame
df = pd.read_csv('dataset_plataformas.csv')

app = FastAPI()

#http://127.0.0.1:8000

# @app.get ('/')
# def prueba():
#     return 'anda bien'

##################################################################################################################################
@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    """
    devuelve sólo el string del nombre de la película con mayor duración según año, plataforma y tipo de duración
    """
    platform = platform.lower()[0]#Aclaro indice en 0 para que tome la primer letra
    duration_type = duration_type.lower()
    #resultado = None
    if (type(year) == int) & (type(platform) == str) & (type(duration_type) == str):
        if (duration_type == 'min') | (duration_type == 'season'):
            resultado = df[(df['release_year']==year) & (df['id'].str.startswith(platform)) & (df['duration_type']== duration_type)]
            idx = resultado['duration_int'].idxmax()
            return {'pelicula': resultado.loc[idx, 'title']}
        else:
            return {'pelicula': 'Intente usar las opciones "min" o "season"'}
    else:
        return {'pelicula': 'Intenta poner valores correctos'}

##################################################################################################################################
################ NO FUNCIONA ###################
@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(platform, scored, year):
    ''' 
    devuelve un int con el total de películas con puntaje mayor a XX en determinado año
    '''
    # cargar el archivo CSV en un DataFrame
    platform = platform.lower()[0]#Aclaro indice en 0 para que tome la primer letra
    resultado = df[(df['release_year']==year) & (df['id'].str.startswith(platform)) & (df['rating_y']> scored) & (df['type'] == 'movie')]
    cantidad = resultado.shape[0]
    return {
            'plataforma': platform,
            'cantidad': cantidad,
            'anio': year,
            'score': float(scored)
            }

##################################################################################################################################

@app.get('/get_count_platform/{plataforma}')
def get_count_platform(platform: str):
    '''
    Devuelve un int con el número total de películas de la plataforma que ingresa
    '''
    #Me aseguro que si el usuario escribe en mayuscula se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = platform.lower()
    platform1 = platform[0] #Aclaro indice en 0 para que tome la primer letra
    #Realizo el filtro para que cuente del dataframe solo la plataforma de la columna ID y movie de la columna type
    cantidad = df[(df['id'].str.startswith(platform1)) & (df['type'] == 'movie')].shape[0]
    return {'plataforma': platform, 'peliculas': cantidad}

##################################################################################################################################
################ NO FUNCIONA ###################

@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(platforma: str, year: int):
    ''' 
    Devuelve sólo el  string con el nombre del actor que más se repite según la plataforma y el año dado
    '''
    # Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = platforma.lower()[0]#Aclaro indice en 0 para que tome la primer letra
    df_filt = df[(df['id'].str.startswith(platform)) & (df['release_year'] == year)]
    # En este paso, se utiliza la función dropna() de pandas para eliminar los valores NaN de la columna "cast" del DataFrame. 
    # Luego, se utiliza el método apply() para aplicar la función str() a cada valor de la columna "cast", convirtiendo los valores a cadenas de caracteres. 
    # Finalmente, se utiliza el método str.split(',') para dividir cada cadena de caracteres en una lista de actores utilizando la coma como separador. 
    # El resultado es una serie de pandas que contiene una lista de actores para cada fila del DataFrame original.
    actores_por_fila = df_filt['cast'].dropna().apply(lambda x: [i.strip() for i in x.split(',') if not i.strip().isdigit()])
    # Cuento la cantidad de veces que aparece cada actor en todas las filas, utilizando la clase Counter de Python.
    contador_actores = Counter()
    for actores in actores_por_fila:
        contador_actores.update(actores)
    # Encuentro el actor que aparece más veces utilizando la funcion most common devolviendo una lista de tuplas donde cada tupla contiene un actor 
    # y la cantidad de veces que aparece en todas las filas del DataFrame.
    actor_mas_repetido = contador_actores.most_common(1)
    if actor_mas_repetido:
        # se asigna [0][0] para indicar el actor que mas veces aparece
        actor_mas_repetido = actor_mas_repetido[0][0]
        # Muestro el actor que aparece más veces y la cantidad de veces que aparece
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

# @app.get('/get_actor/{plataforma}/{anio}')
# def get_actor(platform: str, year: int):
#     df = pd.read_csv('dataset_plataformas.csv')

#     # Filtro por plataforma y año
#     platform = platform.lower()[0]

#     df = df[(df["id"].str.startswith(platform)) & (df["release_year"] == year) & (df["type"] == "movie")]

#     # filtro una lista con todos los actores
#     # Utilizo flatten para obtener un array de numpy unidimensional con todos los valores.
#     # De esta manera, obtengo una lista de todos los actores presentes en el dataframe, sin importar en qué columna aparezcan.
#     actores = df["cast"].str.split(", ", expand=True).values.flatten()

#     # Compruebo si la lista de actores está vacía O tenemos valores en NaN. Si esta condicion se cumple devuelvo "no hay datos".
#     if actores.size == 0 or pd.isnull(actores).all():
#         return "no hay datos"

#     # Contar la cantidad de veces que aparece cada actor
#     cantidad_actores = pd.Series(actores).value_counts()

#     # Obtener el actor más repetido
#     top_actor = cantidad_actores.idxmax()

#     #obtengo el valor de apariciones
#     primer_valor = cantidad_actores.max()
#     # Devolver el nombre del actor
#     return {'plataforma': platform,
#             'anio': year,
#             'actor': top_actor,
#             'apariciones': primer_valor}
##################################################################################################################################

@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    ''' 
    Devuelve el numero total de contenido con ese rating de audiencias
    '''
    rating = rating.lower()
    # Filtrar el dataframe por tipo, país y año
    df_filtrado = df[df['rating'] == rating]
    #me quedo con la cantidad de producciones
    cantidad = len(df_filtrado)
    return {'rating': rating, 'contenido': cantidad}

##################################################################################################################################

@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_country(tipo: str, pais: str, year: int):
    ''' 
    Devuelve el tipo de contenido (pelicula,serie,documental) por pais y año en un diccionario con las variables 
    llamadas 'pais' (nombre del pais), 'anio' (año), 'pelicula' (tipo de contenido)
    '''
    # Filtrar el dataframe por tipo, país y año
    df_filtrado = df[(df['type'] == tipo) & (df['country'] == pais) & (df['release_year'] == year)]
    #me quedo con la cantidad de producciones
    cantidad = len(df_filtrado)
    
    return {'pais': pais,
            'anio': year,
            'apariciones': int(cantidad)
            }

##################################################################################################################################