
# Proyecto individual N°1: Plataformas de Streaming

Primer proyecto de la carrera Data Science de Henry.

## Descripción

Me situo en el rol de un Data Scientist en una empresa que provee servicios de agregación de plataformas de streaming. EL objetivo es desarrollar un modelo de Machine Learning (ML) que elabore un sistema de recomendación de títulos provenientes de las plataformas.
## Principales Tareas

#### Proceso de ETL (Extract, Transform and Load):
Realizo unas pequeñas transformaciones a los datos de los datasets entregados, tanto a los correspondientes de las distintas plataformas como de los ratings de cada usuario. 
Estos fueron:
- Generar un campo id
- Reemplazar valores nulos en la columna rating por el string 'G'
- Pasar las fechas a formato AAAA-mm-dd
- Pasar todos los campos con texto a minúscula
- Convertir el campo duration en duration_int y duration_type

Por último guardo en un dataset ('dataset_plataformas.csv') los datos de las 4 plataformas y sus correspondientes scores

#### Desarrollo de una API:
Particularmente utilizo el framework [FastApi](https://fastapi.tiangolo.com/). Cree 6 endpoints o consultas para consumir en la API:
- Película con mayor duración según año, plataforma y tipo de duración.

        get_max_duration(year, platform, duration_type)
- Cantidad de películas según plataforma, con un puntaje mayor a XX en determinado año.

       get_score_count(platform, scored, year)
- Cantidad de películas según plataforma.

     get_count_platform(platform)
- Actor que más se repite (mayor número de apariciones) según plataforma y año.

        get_actor(platform, year)
- La cantidad de contenidos/productos que se publicó por país y año.

        prod_per_county(tipo,pais,anio)
- La cantidad total de contenidos/productos según el rating de audiencia dado.

        get_contents(rating)
   
#### Deployment:
Hago un deploy en [Render](https://render.com/) para que un usuario externo pueda acceder a mi API.

#### Proceso de EDA (Exploratory Data Analysis):
Vuelvo a realizar unas transformaciones sobre el dataset ('dataset_plataformas.csv') guardado en el proceso de ETL, particularmente en el campo 'rating' para obtener un dataset más limpio y prepararlo para un breve análisis.

#### Sistema de recomendación
El modelo de recomendación se basa en el método de similitud por coseno entre ítems. Consiste en recomendarle al usuario una lista de 5 películas a partir de una película referencia. 

    get_recommendation(titulo: str)




## Aclaraciones

### En el repositorio pueden encontrar:  
- Archivos Jupyter donde realizo el ETL y EDA
- Archivo python donde tengo la app para API con sus consultas  
- Csv de los diferentes datasets utilizados
- Archivo requirements.txt con los módulos utilizados durante el proyecto 

### link del video: 

### drive con datasets iniciales: https://drive.google.com/drive/folders/1yiC_NUYSwR5qBH28g7ztf1yDt_PdqhOS

### Deploy: https://plataformas-deploy.onrender.com/docs#/
