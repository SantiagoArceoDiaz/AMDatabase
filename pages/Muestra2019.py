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
import sklearn as sk
import googletrans
from googletrans import Translator
translator = Translator()

#Bloque para preparar las bases de datos

st.write("# Sobre la muestra") #coloca el titulo de la sección

dfEdades=pd.read_excel('EdadesF.xlsx') # carga el archivo que contiene las edades y nombres de los pacientes

dfEdades['Nombre']= dfEdades['Nombres'] + dfEdades['Apellidos'] #combina la columna de Nombres y la de Apellidos

del dfEdades['Apellidos'] #elimina las filas innecesarias
del dfEdades['Nombres']
del dfEdades['Sexo']

DBEdades=dfEdades[['Nombre', 'Edad']] # Intercambia el orden de las columnas

ListaDBEdades=DBEdades['Nombre'].tolist() #Toma la columna de nombres 
# del archivo de usuarios con edades registradas y crea una lista

SetDBEdades=set(ListaDBEdades) #convierte la lista de usuarios cuya edad está registrada en un conjunto

#carga los datos de los archivos de excel con los resultados de diferentes test para el año 2018
df2020=pd.read_excel('2020C.xlsx')

del df2020['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
del df2020['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

df2019 = df2020.dropna() #quita las filas que tengan NaN en algun valor

df2020['Nombre']= df2020['Nombres'] + df2020['Apellidos'] #combina las columnas de nombres y apellidos en una llamada "Nombre"
del df2020['Apellidos'] # y elimina las columnas individuales.
del df2020['Nombres']
#df2018['Fuerza promedio']=df2018['Prom_Fuer'] 
df2020['Fuerza'] = pd.to_numeric(df2020['Prom_Fuer'])



Listadf2020=df2020['Nombre'].tolist() #crea una lista a partir de los nombres de usuarios en df2018..
Setdf2020=set(Listadf2018) # convierte la lista en un conjunto (para su manejo posterior)

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )


SetDBEdades.difference(Setdf2020) # muestra el conjunto de usuarios que aparecen en la lista de edades
# pero no estan en la base de datos de 2018. Esto puede deberse a que no están o a que se eliminarion por tener columnas con "NaN"

ddf2020 = pd.merge(left=df2020,right=DBEdades, how="inner",on="Nombre")
#ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
# es decir, la intersección vista en el diagrama de Venn.

BD2020=ddf2020[['Nombre','Sexo','Edad', 'MNA', 'Fuerza','Proteinas','BARTHEL', 'Int_BARTHEL']]
#BD2018 # Cambia el orden de las columnas y se guarda como una base de datos nueva.

df=BD2020


# Seleccionar las columnas que quieres filtrar
columnas = ['Edad', 'Sexo', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

# Crear una barra de búsqueda para cada columna en la barra lateral
for col in columnas:
    # Verificar si la columna solo tiene valores de tipo string
    if BD2020[col].dtype == 'object':
        # Obtener los valores únicos en la columna y ordenarlos
        valores = sorted(BD2020[col].unique())
        # Crear una barra de selección para cada valor único en la columna
        seleccion = st.sidebar.multiselect(col, valores, default=valores)
        # Filtrar el dataframe en función de los valores seleccionados en la columna
        BD2020 = BD2020[BD2020[col].isin(seleccion)]
    else:
        # Obtener el rango de valores en la columna
        valores_min = BD2020[col].min()
        valores_max = BD2020[col].max()
        # Crear una barra de selección para el rango de valores en la columna
        seleccion = st.sidebar.slider(col, int(valores_min), int(valores_max), (int(valores_min), int(valores_max)))
        # Filtrar el dataframe en función de los valores seleccionados en la columna
        BD2020 = BD2020[(BD2020[col] >= seleccion[0]) & (BD2020[col] <= seleccion[1])]

st.write(BD2020)


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
venn2020=venn2([Setdf2020, SetDBEdades], set_labels = ('Base de datos de 2018', 'Usuarios con edad registrada'), set_colors=('red','blue'))
st.pyplot(fig)
st.caption("Figura de la comparación entre usuarios en la base de datos de 2018 y usuarios con edad registrada.")


st.markdown(
    """ 
    # Resumen estadistico de la muestra
    Este es un resumen con la estadistica básica de la muestra. Contiene ocho filas que describen estadísticas clave para la base de datos.
    """
    )
Descripcion2020=BD2020.describe() # Crea un resumen con la estadistica de de la base de datos para 2018

st.dataframe(Descripcion2020.style.set_properties(**{'text-align': 'center'}))

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
BD2020Depurada = pd.ExcelWriter('BD2018Depurada.xlsx')
BD2020.to_excel(BD2020Depurada) #convierte BD2018 en un archivo de excel
BD2020Depurada.save() #guarda el archivo de excel en el directorio local
#from google.colab import files
#files.download('BD2018Depurada.xlsx') 




Barras2020, axes = plt.subplots(2, 2, figsize=(10, 10))
#crear un histograma en cada subplot usando "seaborn"
#sns.boxplot(data=df, x='team', y='points', ax=axes[0,0])
#sns.boxplot(data=df, x='team', y='assists', ax=axes[0,1])
sns.histplot(BD2020['Edad'], ax=axes[0,0], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2020['MNA'], ax=axes[0,1], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2020['Fuerza'], ax=axes[1,0], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2020['Proteinas'], ax=axes[1,1], kde=True,
                      line_kws={'linewidth': 2})
st.pyplot(Barras2020)

st.markdown(
    """
    La grafica muestra los histogramas de la distribucion de frecuencias de los paramtero relevantes para la base de datos: Edad [años], Índice Mininutricional [puntaje], Fuerza promedio de antebrazo [kilogramos] y consumo diario de proteinas [gramos]. La línea azul representa una estimación de la densidad de probabilidad de la variable (kde es el acrónimo de "Kernel Density Estimate"). En los histogramas se muestra la distribución de frecuencia de los valores de cada variable. En el eje x se encuentran los valores de la variable y en el eje y se encuentra la frecuencia de los valores.
    """
    )

