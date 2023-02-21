import streamlit as st
import numpy as np
import base64

import pandas as pd #importa la paquetería PAnel DAta (llamada pandas)
#pip install matplotlib_venn
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted # importa paqueteria para graficar diagramas de venn
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt #importa pyplot para hacer gáficas
from matplotlib import numpy as np #importar numpy
import altair as alt
import altair_catplot as altcat
import xlsxwriter
#import os
#import datapane as dp
import matplotlib.pyplot as plt
import seaborn as sns
#import sklearn as sk
import googletrans
from googletrans import Translator
translator = Translator()


st.write("# Sobre la muestra")

#st.title('Antropometria')
#carga los datos de los archivos de excel con los resultados del test de Barthel
dfEdades=pd.read_excel('EdadesF.xlsx')
#dfEdades.head() #muestra las primeras cinco filas de la base de datos

#combina la columna de Nombres y la de Apellidos
dfEdades['Nombre']= dfEdades['Nombres'] + dfEdades['Apellidos'] 

#dfEdades # muestra el dataframe resultante
del dfEdades['Apellidos'] #elimina las filas innecesarias
del dfEdades['Nombres']
del dfEdades['Sexo']
#dfEdades # muestra el dataframe resultante



# Intercambia el orden de las columnas
DBEdades=dfEdades[['Nombre', 'Edad']]
#DBEdades



ListaDBEdades=DBEdades['Nombre'].tolist() #Toma la columna de nombres 
# del archivo de usuarios con edades registradas y crea una lista


SetDBEdades=set(ListaDBEdades) #convierte la lista de usuarios cuya edad está registrada en un conjunto


#carga los datos de los archivos de excel con los resultados de diferentes test para el año 2018
df2018=pd.read_excel('2018C.xlsx')

del df2018['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
del df2018['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

# Se muestra la base depurada, en la que ya se han eliminado aquellos datos con
# NaN, como las columnas PuntajeZ y Marcha tienen solo NaN, entonces se 
# eliminaron, ya que no aportan información al respecto.

df2018 = df2018.dropna() #quita las filas que tengan NaN en algun valor


df2018['Nombre']= df2018['Nombres'] + df2018['Apellidos'] #combina las columnas de nombres y apellidos en una llamada "Nombre"
del df2018['Apellidos'] # y elimina las columnas individuales.
del df2018['Nombres']
#df2018['Fuerza promedio']=df2018['Prom_Fuer'] 
df2018['Fuerza promedio'] = pd.to_numeric(df2018['Prom_Fuer'])

# Cambia el orden de las columnas
#df2018[['Nombre', 'Sexo', 'MNA', 'Prom_Fuer','Proteinas','BARTHEL', 'Int_BARTHEL']]



Listadf2018=df2018['Nombre'].tolist() #crea una lista a partir de los nombres de usuarios en df2018..
Setdf2018=set(Listadf2018) # convierte la lista en un conjunto (para su manejo posterior)

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )


SetDBEdades.difference(Setdf2018) # muestra el conjunto de usuarios que aparecen en la lista de edades
# pero no estan en la base de datos de 2018. Esto puede deberse a que no están o a que se eliminarion por tener columnas con "NaN"

ddf2018 = pd.merge(left=df2018,right=DBEdades, how="inner",on="Nombre")
#ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
# es decir, la intersección vista en el diagrama de Venn.

BD2018=ddf2018[['Nombre','Edad','Sexo', 'MNA', 'Fuerza promedio','Proteinas','BARTHEL', 'Int_BARTHEL']]
#BD2018 # Cambia el orden de las columnas y se guarda como una base de datos nueva.

df=BD2018

# Crear una barra de búsqueda en la barra lateral
#search_term = st.sidebar.text_input('Buscar')

# Filtrar el dataframe en función del término de búsqueda
#if search_term:
#    df = df.query(f"Nombre.str.contains('{search_term}')")

# Mostrar el dataframe filtrado
#st.write(df)

# Seleccionar las columnas que quieres filtrar
#columnas = ['Edad', 'Sexo', 'MNA', 'Prom_Fuer', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

## Crear una barra de búsqueda para cada columna en la barra lateral
##for col in columnas:
#    # Obtener los valores únicos en la columna y ordenarlos
#    valores = sorted(BD2018[col].unique())
#    # Crear una barra de selección para cada valor único en la columna
#    seleccion = st.sidebar.multiselect(col, valores, default=valores)
#    # Filtrar el dataframe en función de los valores seleccionados en la columna
#    BD2018 = BD2018[BD2018[col].isin(seleccion)]

## Cambiar el orden de las columnas y guardar como una base de datos nueva.
#BD2018 = BD2018[['Nombre', 'Edad', 'Sexo', 'MNA', 'Prom_Fuer', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]

## Crear una barra de búsqueda en la barra lateral
#search_term = st.sidebar.text_input('Buscar')

# Filtrar el dataframe en función del término de búsqueda
#if search_term:
#    BD2018 = BD2018[BD2018['Nombre'].str.contains(search_term)]

# Mostrar el dataframe filtrado
#st.write(BD2018)

# Seleccionar las columnas que quieres filtrar
#columnas = ['Edad', 'Sexo', 'MNA', 'Fuerza promedio', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

## Crear una barra de búsqueda para cada columna en la barra lateral
#for col in columnas:
#    # Obtener el rango de valores en la columna
#    valores_min = BD2018[col].min()
#    valores_max = BD2018[col].max()
#    # Crear una barra de selección para el rango de valores en la columna
#    seleccion = st.sidebar.slider(col, valores_min, valores_max, (valores_min, valores_max))
#    # Filtrar el dataframe en función de los valores seleccionados en la columna
#    BD2018 = BD2018[(BD2018[col] >= seleccion[0]) & (BD2018[col] <= seleccion[1])]

## Cambiar el orden de las columnas y guardar como una base de datos nueva.
#BD2018 = BD2018[['Nombre', 'Edad', 'Sexo', 'MNA', 'Prom_Fuer', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]

# Crear una barra de búsqueda en la barra lateral
#search_term = st.sidebar.text_input('Buscar')

# Filtrar el dataframe en función del término de búsqueda
#if search_term:
#    BD2018 = BD2018[BD2018['Nombre'].str.contains(search_term)]

# Mostrar el dataframe filtrado
#st.write(BD2018)

# Seleccionar las columnas que quieres filtrar
columnas = ['Edad', 'Sexo', 'MNA', 'Fuerza promedio', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

# Crear una barra de búsqueda para cada columna en la barra lateral
for col in columnas:
    # Verificar si la columna solo tiene valores de tipo string
    if BD2018[col].dtype == 'object':
        # Obtener los valores únicos en la columna y ordenarlos
        valores = sorted(BD2018[col].unique())
        # Crear una barra de selección para cada valor único en la columna
        seleccion = st.sidebar.multiselect(col, valores, default=valores)
        # Filtrar el dataframe en función de los valores seleccionados en la columna
        BD2018 = BD2018[BD2018[col].isin(seleccion)]
    else:
        # Obtener el rango de valores en la columna
        valores_min = BD2018[col].min()
        valores_max = BD2018[col].max()
        # Crear una barra de selección para el rango de valores en la columna
        seleccion = st.sidebar.slider(col, int(valores_min), int(valores_max), (int(valores_min), int(valores_max)))
        # Filtrar el dataframe en función de los valores seleccionados en la columna
        BD2018 = BD2018[(BD2018[col] >= seleccion[0]) & (BD2018[col] <= seleccion[1])]

st.write(BD2018)


# Cambiar el orden de las columnas y guardar como una base de datos nueva.
#BD2018 = BD2018[['Nombre', 'Edad', 'Sexo', 'MNA', 'Fuerza promedio', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]

# Crear una barra de búsqueda en la barra lateral
#search_term = st.sidebar.text_input('Buscar')

# Filtrar el dataframe en función del término de búsqueda
#if search_term:
#    BD2018 = BD2018[BD2018['Nombre'].str.contains(search_term)]

# Mostrar el dataframe filtrado

#st.write(print(BD2018.dtypes))





# Crear un botón de descarga para el dataframe
def download_button_CSV(df, filename, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Crear un botón de descarga para el dataframe
def download_button(df, filename, button_text):
    # Crear un objeto ExcelWriter
    excel_writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # Guardar el dataframe en el objeto ExcelWriter
    df.to_excel(excel_writer, index=False)
    # Cerrar el objeto ExcelWriter
    excel_writer.save()
    # Leer el archivo guardado como bytes
    with open(filename, 'rb') as f:
        file_bytes = f.read()
    # Generar el enlace de descarga
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_bytes).decode()}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(df, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(df, 'dataframe.csv', 'Descargar como CSV')
    st.write('')

st.markdown(
    """ 
    # Diagrama de venn 👋
    La muestra de recolectada en 2018 representa compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

# crea un diagrama de Venn en donde podemos ver los usuarios que tienen en común la base de datos de 2018 y la de edades registradas
fig, ax = plt.subplots(figsize=(2,2))
venn2018=venn2([Setdf2018, SetDBEdades], set_labels = ('Base de datos de 2018', 'Usuarios con edad registrada'), set_colors=('red','blue'))
st.pyplot(fig)
st.caption("Figura de la comparación entre usuarios en la base de datos de 2018 y usuarios con edad registrada.")


st.markdown(
    """ 
    # Resumen estadistico de la muestra
    Este es un resumen con la estadistica básica de la muestra. Contiene ocho filas que describen estadísticas clave para la base de datos.
    """
    )
Descripcion2018=BD2018.describe() # Crea un resumen con la estadistica de de la base de datos para 2018

st.dataframe(Descripcion2018.style.set_properties(**{'text-align': 'center'}))

st.markdown(
    """
    Las filas son las siguientes:

    - **count:** el número de valores no nulos en la columna.
    - **mean:** la media aritmética de los valores en la columna.
    - **std:** la desviación estándar de los valores en la columna.
    - **min:** el valor mínimo de la columna.
    - **25%:** el primer cuartil de la columna, es decir, el valor que separa el 25% inferior de los valores de la columna del 75% superior.
    - **50%:** la mediana de la columna, es decir, el valor que separa el 50% inferior de los valores de la columna del 50% superior.
    - **75%:** el tercer cuartil de la columna, es decir, el valor que separa el 75% inferior de los valores de la columna del 25% superior.
    - **max:** el valor máximo de la columna.
    """
    )


# Escribe un archivo de excel que contine a BD2018
BD2018Depurada = pd.ExcelWriter('BD2018Depurada.xlsx')
BD2018.to_excel(BD2018Depurada) #convierte BD2018 en un archivo de excel
BD2018Depurada.save() #guarda el archivo de excel en el directorio local
#from google.colab import files
#files.download('BD2018Depurada.xlsx') 




Barras2018, axes = plt.subplots(2, 2, figsize=(10, 10))
#crear un histograma en cada subplot usando "seaborn"
#sns.boxplot(data=df, x='team', y='points', ax=axes[0,0])
#sns.boxplot(data=df, x='team', y='assists', ax=axes[0,1])
sns.histplot(BD2018['Edad'], ax=axes[0,0], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2018['MNA'], ax=axes[0,1], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2018['Fuerza promedio'], ax=axes[1,0], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(df2018['Proteinas'], ax=axes[1,1], kde=True,
                      line_kws={'linewidth': 2})
st.pyplot(Barras2018)

st.markdown(
    """
    La grafica muestra los histogramas de la distribucion de frecuencias de los paramtero relevantes para la base de datos: Edad [años], Índice Mininutricional [puntaje], Fuerza promedio de antebrazo [kilogramos] y consumo diario de proteinas [gramos]. La línea azul representa una estimación de la densidad de probabilidad de la variable (kde es el acrónimo de "Kernel Density Estimate"). En los histogramas se muestra la distribución de frecuencia de los valores de cada variable. En el eje x se encuentran los valores de la variable y en el eje y se encuentra la frecuencia de los valores.
    """
    )





chart1=altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('MNA:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza promedio'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

chart2=altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Proteinas:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza promedio'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

#chart3=altcat.catplot(BD2018,
#               height=350,
#               width=450,
#               mark='point',
#               box_mark=dict(strokeWidth=2, opacity=0.6),
#               whisker_mark=dict(strokeWidth=2, opacity=0.9),
#               encoding=dict(x=alt.X('Sexo:N', title=None),
#                             y=alt.Y('PuntajeZ:Q',scale=alt.Scale(zero=False)),
#                             tooltip=[alt.Tooltip("Nombre Completo"),
#                             alt.Tooltip("Edad"),
#                             alt.Tooltip("Proteinas"),
#                             alt.Tooltip("Prom_Fuer"),
#                             alt.Tooltip("MNA"),
#                             #alt.Tooltip("Marcha"),
#                             alt.Tooltip("BARTHEL"),
#                             ],
#                             color=alt.Color('Sexo:N', legend=None)),
#               transform='jitterbox',
#              jitter_width=0.5).interactive()

chart4=altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Fuerza promedio:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza promedio'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

#chart5=altcat.catplot(BD2018,
#               height=350,
#               width=450,
#               mark='point',
#               box_mark=dict(strokeWidth=2, opacity=0.6),
#               whisker_mark=dict(strokeWidth=2, opacity=0.9),
#               encoding=dict(x=alt.X('Sexo:N', title=None),
#                             y=alt.Y('Marcha:Q',scale=alt.Scale(zero=False)),
#                             tooltip=[alt.Tooltip("Nombre Completo"),
#                             alt.Tooltip("Edad"),
#                             alt.Tooltip("Proteinas"),
#                             alt.Tooltip("Prom_Fuer"),
#                             alt.Tooltip("MNA"),
#                             #alt.Tooltip("Marcha"),
#                             alt.Tooltip("BARTHEL"),
#                             ],
#                             color=alt.Color('Sexo:N', legend=None)),
#               transform='jitterbox',
#              jitter_width=0.5).interactive()

chart6=altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Fuerza promedio:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza promedio'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


cajas2018=alt.vconcat(alt.hconcat(chart1, chart2),alt.hconcat(chart4, chart6))

#Barras2018, axes = plt.subplots(2, 2, figsize=(10, 10))
#crear un histograma en cada subplot usando "seaborn"
#sns.boxplot(data=df, x='team', y='points', ax=axes[0,0])
#sns.boxplot(data=df, x='team', y='assists', ax=axes[0,1])
#sn.histplot(BD2018['Edad'], ax=axes[0,0], kde=True,
#                      line_kws={'linewidth': 2})
#sn.histplot(BD2018['MNA'], ax=axes[0,1], kde=True,
#                      line_kws={'linewidth': 2})
#sn.histplot(BD2018['Prom_Fuer'], ax=axes[1,0], kde=True,
#                      line_kws={'linewidth': 2})
#sn.histplot(df2018['Proteinas'], ax=axes[1,1], kde=True,
#                      line_kws={'linewidth': 2})
st.altair_chart(cajas2018)


selection = alt.selection_multi(fields=['Sexo'], bind='legend')
chart1 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='Fuerza promedio',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza promedio'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=400, width=500
).add_selection(
    selection
).interactive()

chart2 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='Proteinas',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza promedio'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=400, width=500
).add_selection(
    selection
).interactive()



chart3 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='MNA',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza promedio'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=400, width=500
).add_selection(
    selection
).interactive()

#chart4 = alt.Chart(BD2018).mark_circle(size=50).encode(
#    x='Edad', y='Marcha',
#    color='Sexo',
#    tooltip=[alt.Tooltip("Nombre Completo"),
#    alt.Tooltip("MNA"),
#    alt.Tooltip("Prom_Fuer"),
#    #alt.Tooltip("Marcha"),
#    #alt.Tooltip("PuntajeZ"),
#    alt.Tooltip("BARTHEL"),
#    ],
#    opacity=alt.condition(selection, alt.value(1), alt.value(0))
#).properties(
#    height=400, width=500
#).add_selection(
#    selection
#).interactive()

#chart5 = alt.Chart(BD2018).mark_circle(size=50).encode(
#    x='Edad', y='PuntajeZ',
#    color='Sexo',
#    tooltip=[alt.Tooltip("Nombre Completo"),
#    alt.Tooltip("MNA"),
#    alt.Tooltip("Prom_Fuer"),
#    #alt.Tooltip("Marcha"),
#    #alt.Tooltip("PuntajeZ"),
#    alt.Tooltip("BARTHEL"),
#    ],
#    opacity=alt.condition(selection, alt.value(1), alt.value(0))
#).properties(
#    height=400, width=500
#).add_selection(
#    selection
#).interactive()

selection = alt.selection_multi(fields=['Sexo'], bind='legend')
chart6 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='BARTHEL',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza promedio'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=400, width=500
).add_selection(
    selection
).interactive()


correlaciones2018=alt.vconcat(alt.hconcat(chart1, chart2),alt.hconcat(chart3, chart6))

st.altair_chart(correlaciones2018)

#Barras2018, axes = plt.subplots(2, 2, figsize=(10, 10))


# Creamos una correlación desde un dataset D
#corr2018 = BD2018.corr()

# Dibujamos nuestro gráfico
#grafico=sns.heatmap(corr2018)
#st.pyplot(grafico)


#datos = sns.load_dataset("iris")
#grafico = sns.pairplot(datos, hue="species")

#st.pyplot(grafico)

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

# localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Masculino."
Hombres2018=BD2018.loc[BD2018['Sexo']=="Mas"]
del Hombres2018['Sexo'] #Borra la columna de "Sexo", ya que es innecesaria
Hombres2018 # Muestra el dataframe con datos de hombres.


Hombres2018.describe() # Crea un resumen estadistico sobre el dataframe "Hombres 2018".

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

Mujeres2018=BD2018.loc[BD2018['Sexo']=="Fem"] # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Femenino."

del Mujeres2018['Sexo']
Mujeres2018



Mujeres2018.describe() # dEscripcion del Dataframe de "Mujeres"


# En el siguiente bloque se separa por rangos de edad de 10 años al subgrupo de "Hombres". 
# En algunos casos no hay miembros registrados que cumplan con el rango de edad requerido, por lo que los
# dataframes estan vacios (como ocurre con el grupo de menos de 60 años).

Hombres201860=BD2018.loc[((BD2018['Edad'] <= 60) & (BD2018['Sexo']=='Mas'))]
del Hombres201860['Sexo']
Hombres201870=BD2018.loc[((BD2018['Edad'] > 60) & (BD2018['Edad'] <= 70) & (BD2018['Sexo'] == 'Mas'))]
del Hombres201870['Sexo']
Hombres201880=BD2018.loc[((BD2018['Edad'] > 70) & (BD2018['Edad'] <= 80) & (BD2018['Sexo'] == 'Mas'))]
del Hombres201880['Sexo']
Hombres201890=BD2018.loc[((BD2018['Edad'] > 80) & (BD2018['Edad'] <= 90) & (BD2018['Sexo'] == 'Mas'))]
del Hombres201890['Sexo']
Hombres2018100=BD2018.loc[((BD2018['Edad'] > 90) & (BD2018['Sexo'] == 'Mas'))]
del Hombres2018100['Sexo']



#Hombres201860
#Hombres201870
#Hombres201880
#Hombres201890
#Hombres2018100



Mujeres201860=BD2018.loc[((BD2018['Edad']<=60) & (BD2018['Sexo']=='Fem'))]
del Mujeres201860['Sexo']
Mujeres201870=BD2018.loc[((BD2018['Edad'] >60) & (BD2018['Edad']<=70) & (BD2018['Sexo']=='Fem'))]
del Mujeres201870['Sexo']
Mujeres201880=BD2018.loc[((BD2018['Edad'] >70) & (BD2018['Edad']<=80) & (BD2018['Sexo']=='Fem'))]
del Mujeres201880['Sexo']
Mujeres201890=BD2018.loc[((BD2018['Edad'] >80) & (BD2018['Edad']<=90) & (BD2018['Sexo']=='Fem'))]
del Mujeres201890['Sexo']
Mujeres2018100=BD2018.loc[((BD2018['Edad'] >90) & (BD2018['Sexo']=='Fem'))]
del Mujeres2018100['Sexo']



#Mujeres201860
#Mujeres201870
#Mujeres201880
#Mujeres201890
#Mujeres2018100



#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
# Creamos una correlación desde un dataset D
corr = Hombres201870.corr().loc[:'BARTHEL', :"BARTHEL"]

# Dibujamos nuestro gráfico
sns.heatmap(corr)
plt.show()
#st.seaborn(Barras2018)



#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
# Creamos una correlación desde un dataset D
corr = Hombres201880.corr().loc[:'BARTHEL', :"BARTHEL"]

# Dibujamos nuestro gráfico
sns.heatmap(corr)
plt.show()



# importing libraries
import altair as alt
from vega_datasets import data
  
# importing airports dataset from vega_datasets package
# airport = data.airports()
  
# making the scatter plot on latitude and longitude
# setting color on the basis of country
fig = alt.Chart(Hombres201870).mark_point().encode(
  x='Edad',y='Proteinas')
  
# making the regression line using transform_regression
# function and add with the scatter plot
final_plot = fig + fig.transform_regression('latitude','longitude').mark_line()
  
# saving the scatter plot with regression line
final_plot.save('output2.html')



# Creamos una correlación desde un dataset D
corr = Hombres201880.corr().loc[:'BARTHEL', :"BARTHEL"]

# Dibujamos nuestro gráfico
sns.heatmap(corr)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Creamos una correlación desde un dataset D
corr = Hombres201890.corr().loc[:'BARTHEL', :"BARTHEL"]

# Dibujamos nuestro gráfico
sns.heatmap(corr)
plt.show()




#import seaborn as sns
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
#sb1 = sns.heatmap(
#    subset1.corr(), 
#    cmap = cmap,
#    square=True, 
#    cbar_kws={ 'shrink' : .9 }, 
#    annot = True, 
#    annot_kws = { 'fontsize' : 12 })

# Here we create a figure instance, and two subplots
CalorHombres2018 = plt.figure(figsize = (20,20)) # width x height
ax1 = CalorHombres2018.add_subplot(2, 2, 1) # row, column, position
ax2 = CalorHombres2018.add_subplot(2, 2, 2)
ax3 = CalorHombres2018.add_subplot(2, 2, 3)


# We use ax parameter to tell seaborn which subplot to use for this plot
sns.heatmap(data=Hombres201870.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax1, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
sns.heatmap(data=Hombres201880.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax2, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
sns.heatmap(data=Hombres201890.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax3, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})






#import seaborn as sns
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
#sb1 = sns.heatmap(
#    subset1.corr(), 
#    cmap = cmap,
#    square=True, 
#    cbar_kws={ 'shrink' : .9 }, 
#    annot = True, 
#    annot_kws = { 'fontsize' : 12 })

# Here we create a figure instance, and two subplots
CalorMujeres2018 = plt.figure(figsize = (20,20)) # width x height
ax1 = CalorMujeres2018.add_subplot(2, 2, 1) # row, column, position
ax2 = CalorMujeres2018.add_subplot(2, 2, 2)
ax3 = CalorMujeres2018.add_subplot(2, 2, 3)
ax4 = CalorMujeres2018.add_subplot(2, 2, 4)


# We use ax parameter to tell seaborn which subplot to use for this plot
sns.heatmap(data=Mujeres201860.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax1, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
sns.heatmap(data=Mujeres201870.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax2, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
sns.heatmap(data=Mujeres201880.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax3, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
sns.heatmap(data=Mujeres201890.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax4, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
st.pyplot(CalorMujeres2018)

