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
df2018=pd.read_excel('2018C.xlsx')

del df2018['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
del df2018['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

df2018 = df2018.dropna() #quita las filas que tengan NaN en algun valor

df2018['Nombre']= df2018['Nombres'] + df2018['Apellidos'] #combina las columnas de nombres y apellidos en una llamada "Nombre"
del df2018['Apellidos'] # y elimina las columnas individuales.
del df2018['Nombres']
#df2018['Fuerza promedio']=df2018['Prom_Fuer'] 
df2018['Fuerza'] = pd.to_numeric(df2018['Prom_Fuer'])



Listadf2018=df2018['Nombre'].tolist() #crea una lista a partir de los nombres de usuarios en df2018..
Setdf2018=set(Listadf2018) # convierte la lista en un conjunto (para su manejo posterior)

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 263 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. Sin embargo, en muchos casos, la información recolectada presentaba registros incompletos. **La muestra depurada es de 152 indiviudos**. Para todos los individuos en la muestra depurada se cuenta datoso sobre: **sexo, edad, índice mininutricional, fuerza de brazo, contenido de proteinas en sangre e índice de Barthel**. A continuación se muestra la base de datos de los participantes. 
    """
    )


SetDBEdades.difference(Setdf2018) # muestra el conjunto de usuarios que aparecen en la lista de edades
# pero no estan en la base de datos de 2018. Esto puede deberse a que no están o a que se eliminarion por tener columnas con "NaN"

ddf2018 = pd.merge(left=df2018,right=DBEdades, how="inner",on="Nombre")
#ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
# es decir, la intersección vista en el diagrama de Venn.

BD2018=ddf2018[['Nombre','Sexo','Edad', 'MNA', 'Fuerza','Proteinas','BARTHEL', 'Int_BARTHEL']]
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
columnas = ['Edad', 'Sexo', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

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
    # Diagrama de venn 
    Representativamente, **la muestra depurada de 2018, representa el 58 por ciento** del total de pacientes registrados ese año.  
    """
    )

# crea un diagrama de Venn en donde podemos ver los usuarios que tienen en común la base de datos de 2018 y la de edades registradas
fig, ax = plt.subplots(figsize=(2,2))
venn2018=venn2([Setdf2018, SetDBEdades], set_labels = ('Muestra depurada', 'Muestra total de 2018'), set_colors=('red','blue'))
st.pyplot(fig)
st.caption("**Figura 1:** Diagrama de venn en la que se muestra el tamaño relativo de la muestra depurada, en comparación con el total de pacientes registrados en 2018.")


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

BD2018cat=BD2018.copy()
### Categorizacion

# Definimos una función para aplicar a cada valor de la columna MNA
def categorizar_mna(valor):
    if valor < 17:
        return 0
    elif valor >= 17 and valor <= 23.5:
        return 1
    else:
        return 2

def categorizar_fza(valor):
    if valor < 17:
        return 0
    elif valor >= 17 and valor <= 23.5:
        return 1
    else:
        return 2

# Definimos una función para aplicar a cada valor de la columna MNA y genero
def categorizar_fza_genero(valor, sexo):
    if sexo == 'Mas':
        if valor > 29.99:
            return 0
        else:
            return 1
    elif sexo == 'Fem':
        if valor > 19.99:
            return 0
        else:
            return 1



# Aplicamos la función a cada valor de la columna MNA utilizando apply()
BD2018cat['MNA'] = BD2018cat['MNA'].apply(categorizar_mna)
BD2018cat['Fuerza'] = BD2018cat.apply(lambda x: categorizar_fza_genero(x['Fuerza'], x['Sexo']), axis=1)
#BD2018cat
# Mostramos el DataFrame resultante
#print(df)

###

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
sns.histplot(BD2018['Fuerza'], ax=axes[1,0], kde=True,
                      line_kws={'linewidth': 2})
sns.histplot(BD2018['Proteinas'], ax=axes[1,1], kde=True,
                      line_kws={'linewidth': 2})
st.pyplot(Barras2018)

st.markdown(
    """
    La grafica muestra los histogramas de la distribucion de frecuencias de los paramtero relevantes para la base de datos: Edad [años], Índice Mininutricional [puntaje], Fuerza promedio de antebrazo [kilogramos] y consumo diario de proteinas [gramos]. La línea azul representa una estimación de la densidad de probabilidad de la variable (kde es el acrónimo de "Kernel Density Estimate"). En los histogramas se muestra la distribución de frecuencia de los valores de cada variable. En el eje x se encuentran los valores de la variable y en el eje y se encuentra la frecuencia de los valores.
    """
    )

#alt.themes.set_config(width='30em')


chart1=altcat.catplot(BD2018,
               height=400,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('MNA:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

chart2=altcat.catplot(BD2018,
               height=400,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Proteinas:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


chart4=altcat.catplot(BD2018,
               height=400,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Fuerza:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()



chart6=altcat.catplot(BD2018,
               height=400,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Fuerza:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza'),
                             alt.Tooltip("MNA"),
                             #alt.Tooltip("Marcha"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


cajas2018=alt.vconcat(alt.hconcat(chart1, chart2),alt.hconcat(chart4, chart6))
st.altair_chart(cajas2018)


selection = alt.selection_multi(fields=['Sexo'], bind='legend')
chart1 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='Fuerza',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=350, width=450
).add_selection(
    selection
).interactive()

chart2 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='Proteinas',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=350, width=450
).add_selection(
    selection
).interactive()



chart3 = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad', y='MNA',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre"),
    alt.Tooltip("MNA"),
    alt.Tooltip('Fuerza'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=350, width=450
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
    alt.Tooltip('Fuerza'),
    #alt.Tooltip("Marcha"),
    #alt.Tooltip("PuntajeZ"),
    alt.Tooltip("BARTHEL"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=350, width=450
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
    # Sub muestra de hombres
    A continuación se muestra la base de datos de los participantes de género masculino
    """
    )

# localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Masculino."
Hombres2018=BD2018.loc[BD2018['Sexo']=="Mas"]
del Hombres2018['Sexo'] #Borra la columna de "Sexo", ya que es innecesaria
Hombres2018 # Muestra el dataframe con datos de hombres.

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Hombres2018, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Hombres2018, 'dataframe.csv', 'Descargar como CSV')
    st.write('')


st.markdown(
    """ 
    Resumen estadístico de la sub-muestra de hombres 
    """
    )

DescripcionHombres2018=Hombres2018.describe() # Crea un resumen con la estadistica de de la base de datos para 2018

st.dataframe(DescripcionHombres2018.style.set_properties(**{'text-align': 'center'}))

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

st.markdown(
    """ 
    Hombres menores de 70 años
    A continuación se muestra la base de datos de los participantes de género femenino.
    """
    )
Hombres201870

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Hombres201870, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Hombres201870, 'dataframe.csv', 'Descargar como CSV')
    st.write('')


st.markdown(
    """ 
    Hombres menores de 80 años
    A continuación se muestra la base de datos de los participantes de género femenino.
    """
    )

Hombres201880

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Hombres201880, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Hombres201880, 'dataframe.csv', 'Descargar como CSV')
    st.write('')

st.markdown(
    """ 
    Hombres menores de 90 años
    A continuación se muestra la base de datos de los participantes de género femenino.
    """
    )
Hombres201890

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Hombres201890, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Hombres201890, 'dataframe.csv', 'Descargar como CSV')
    st.write('')



st.markdown(
    """ 
    Mapas de calor
    """
    )


corr = Hombres201870.corr().loc[:'BARTHEL', :"BARTHEL"]
sns.heatmap(corr)
plt.show()


corr = Hombres201880.corr().loc[:'BARTHEL', :"BARTHEL"]
sns.heatmap(corr)
plt.show()


corr = Hombres201890.corr().loc[:'BARTHEL', :"BARTHEL"]
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
st.pyplot(CalorHombres2018)



st.markdown(
    """ 
    # Sub muestra de mujeres
    A continuación se muestra la base de datos de los participantes de género femenino.
    """
    )

Mujeres2018=BD2018.loc[BD2018['Sexo']=="Fem"] # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Femenino."

del Mujeres2018['Sexo']
Mujeres2018

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Mujeres2018, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Mujeres2018, 'dataframe.csv', 'Descargar como CSV')
    st.write('')


st.markdown(
    """ 
    Resumen estadístico de la sub-muestra de mujeres
    """
    )

DescripcionMujeres2018=Mujeres2018.describe() # Crea un resumen con la estadistica de de la base de datos para 2018

st.dataframe(DescripcionMujeres2018.style.set_properties(**{'text-align': 'center'}))


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


st.markdown(
    """ 
    Mujeres menores de 60 años
    """
    )
Mujeres201860
# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Mujeres201860, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Mujeres201860, 'dataframe.csv', 'Descargar como CSV')
    st.write('')


st.markdown(""" 
    Mujeres menores de 70 años
    """)
Mujeres201870
# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Mujeres201870, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Mujeres201870, 'dataframe.csv', 'Descargar como CSV')
    st.write('')



st.markdown(
    """ 
    Mujeres menores de 80 años
    """
    )
Mujeres201880
# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Mujeres201880, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Mujeres201880, 'dataframe.csv', 'Descargar como CSV')
    st.write('')



st.markdown(
    """ 
    Mujeres menores de 90 años
    """
    )
Mujeres201890

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Agregar un botón de descarga para el dataframe en la primera columna
with col1:
    download_button(Mujeres201890, 'dataframe.xlsx', 'Descargar como Excel')
    st.write('')

# Agregar un botón de descarga para el dataframe en la segunda columna
with col2:
    download_button_CSV(Mujeres201890, 'dataframe.csv', 'Descargar como CSV')
    st.write('')


st.markdown(
    """ 
    Mapas de calor
    """)



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

st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes values 
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X,y,learningrate, iterations):
    """ Find the multivarite regression model for the data set
         Parameters:
          X:  independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          iterations: the number of iterations
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1))   
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y))*((y_pred - y)**2) #Calculate the loss for each training instance
        total = 0
        for i in range(len(y)):
            total += cost_value[i][0] #Calculate the cost function for each iteration
        cost_lst.append(total)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    ax.set_title('Cost function Graph')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost')
    st.pyplot(fig)
    return theta


BD2018 = BD2018[['Nombre','Sexo', 'Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]
X = BD2018.iloc[:,2:-2].values
y = BD2018.iloc[:,-2].values

sc=StandardScaler()
X_transform=sc.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_transform, y)
lin_reg.intercept_, lin_reg.coef_

# Find the optimal theta values using the custom function
theta_optimal = Multivariable_Linear_Regression(X_transform, y, 0.03, 30000)

# Create a new dataframe with the original data and predicted values
X_transform_df = pd.DataFrame(X_transform, columns=['Edad', 'MNA', 'Fuerza', 'Proteinas'])
predictions = np.dot(X_transform_df, theta_optimal[1:]) + theta_optimal[0]
BD2018_with_predictions = BD2018.assign(Predicted_BARTHEL=predictions)

# Print the new dataframe with predictions
st.write(BD2018_with_predictions)


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes values 
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X,y,learningrate, iterations):
    """ Find the multivarite regression model for the data set
         Parameters:
          X:  independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          iterations: the number of iterations
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1))   
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y))*((y_pred - y)**2) #Calculate the loss for each training instance
        total = 0
        for i in range(len(y)):
            total += cost_value[i][0] #Calculate the cost function for each iteration
        cost_lst.append(total)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    ax.set_title('Cost function Graph')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost')
    st.pyplot(fig)
    return theta


BD2018 = BD2018[['Nombre','Sexo', 'Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]
X = BD2018.iloc[:,2:-2].values
y = BD2018.iloc[:,-2].values

sc=StandardScaler()
X_transform=sc.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_transform, y)
lin_reg.intercept_, lin_reg.coef_

# Find the optimal theta values using the custom function
theta_optimal = Multivariable_Linear_Regression(X_transform, y, 0.03, 30000)

# Create a new dataframe with the original data and predicted values
X_transform_df = pd.DataFrame(X_transform, columns=['Edad', 'MNA', 'Fuerza', 'Proteinas'])
predictions = np.dot(X_transform_df, theta_optimal[1:]) + theta_optimal[0]
BD2018_with_predictions = BD2018.assign(Predicted_BARTHEL=predictions)

# Print the new dataframe with predictions
st.write(BD2018_with_predictions)


from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree


# Cargar los datos del conjunto iris
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#colnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'name']
#iris = pd.read_csv(url, header=None, names=colnames)

# Convertir los datos a un arreglo numpy
#X = iris.iloc[:, :4].values
#y = iris.iloc[:, 4].values
#BD2018["Fuerza"]=BD2018["Prom_Fuer"]
BD2018 = BD2018[['Nombre','Sexo', 'Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]
#X = BD2018.iloc[:,2:-2].values
#y = BD2018.iloc[:,-2].values

X = BD2018.iloc[:,2:-2].values
y = BD2018.iloc[:,-1].values


# Definir el algoritmo de clasificación y ajustar los datos
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)

# Pedir al usuario los valores de cada atributo
#Edad = float(input("Introduzca la Edad: "))
#MNA = float(input("Introduzca el resultado del test MNA: "))
#Fuerza = float(input("Introduzca el promedio de fuerza de presión: "))
#Proteinas = float(input("Introduzca el consumo promedio de proteinas: "))

# Crear los deslizadores
Edad = st.slider("Edad", 60, 100, 75)
MNA = st.slider("MNA", 0, 30, 15)
Fuerza = st.slider("Fuerza", 0, 150, 75)
Proteinas = st.slider("Proteinas", 0, 200, 100)

# Clasificar el objeto
prediction = clf.predict([[Edad, MNA, Fuerza, Proteinas]])
print("El objeto pertenece a la clase:", prediction[0])


#from sklearn.datasets import load_iris
#from sklearn import tree
#import numpy as np

# Load the iris dataset
#iris = load_iris()

# Split the dataset into training and testing datasets
#train_data = iris.data[:-20]
#train_data = BD2018.iloc[:-20, :]
train_data = BD2018.iloc[:-20,2:-2].values

#train_target = iris.target[:-20]
train_target = BD2018.iloc[:-20, -1].values

#test_data = iris.data[-20:]
test_data = BD2018.iloc[-20:].values


#test_target = iris.target[-20:]
test_target = BD2018.iloc[-20:, -1].values

# Train a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# Use the trained classifier to classify a new object
new_object = np.array([[Edad, MNA, Fuerza, Proteinas]])
prediction = clf.predict(new_object)

# Print the preditrain_target.shapection
#print("The predicted class is:", iris.target_names[prediction[0]])

clf = DecisionTreeClassifier()
clf.fit(X, y)


class_names=BD2018.columns[-1] #
#tree_rules = export_text(clf, feature_names=BD2018.columns[2:-2])
#tree_rules = export_text(clf, feature_names=BD2018.columns[2:-2].tolist()), class_names=BD2018.columns[-1]
tree_rules = sk.tree.export_text(clf, feature_names=BD2018.columns[2:-2].tolist())

st.text(tree_rules)

#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
#tree.plot_tree(clf)
#st.pyplot(fig)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(clf, filled=True, feature_names=BD2018.columns[2:-2].tolist(), class_names=BD2018.columns[-1])
plt.show()
st.pyplot(fig)

from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    st.write("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            st.write("{}if {} <= {}:".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            st.write("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            st.write("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

#class_names = BD2018['target'].unique().astype(str)#fff
#class_names = BD2018.columns[-1].unique().astype(str)
class_names = ['0', '1', '2']
rules = get_rules(clf, BD2018.columns[2:-2].tolist(), BD2018.columns[-1])
for r in rules:
    st.write(r)

#sklearn.tree.export_text(


#import graphviz 
#os.environ["PATH"] += os.pathsep + 'C:/Users/Santiago/anaconda3/Lib/site-packages/graphviz/dot.py'
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris")
#st.graphviz_chart(graph.render(engine='dot'), use_container_width=True)

#dot_data = tree.export_graphviz(clf, out_file=None, 
                      #feature_names=iris.feature_names,  
#                      feature_names=BD2018.columns[2:-1],
                      #class_names=iris.target_names,  
#                      class_names=BD2018.columns[-1],
#                      filled=True, rounded=True,  
#                      special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph 

#from sklearn import tree
#import streamlit as st
#import pydotplus
#from streamlit_pydotplus import st_pydotplus
#from IPython.display import Image  

#features =[[1,0,0] , [2,1,1] , [3,0,0] , [4,1,1] , [5,0,0] , [6,0,1] ,  [900,0,1] , [1001,0,0]] #val,pow2,even
#labels =  ['o'     ,  'e'    , 'o'     ,  'e'    ,  'o'    ,  'e'    ,   'e'      ,  'o'] #is even

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(features,labels)

#prediction = clf.predict([[203,0,0]])

#st.write(f"Prediction: {prediction[0]}")

#dot_data = tree.export_graphviz(clf, out_file=None, 
#                     feature_names=['number','pow2','even'],  
#                     class_names=['o','e'],  
#                     filled=True, rounded=True,  
#                     special_characters=True)  
#graph = pydotplus.graph_from_dot_data(dot_data)  

#st_pydotplus(graph)

#import streamlit as st
#from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier, plot_tree
#import matplotlib.pyplot as plt
#import numpy as np

# load iris dataset
#iris = load_iris()

# fit decision tree classifier
#clf = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)

# create a meshgrid to plot the decision surface
#x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:, 0].max() + 1
#y_min, y_max = iris.data[:, 1].min() - 1, iris.data[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                     np.arange(y_min, y_max, 0.02))

# predict on the meshgrid
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)

# plot the decision surface
#fig, ax = plt.subplots()
#ax.contourf(xx, yy, Z, alpha=0.4)
#ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, alpha=0.8)
#ax.set_xlabel('Sepal length')
#ax.set_ylabel('Sepal width')
#ax.set_title('Decision surface of a decision tree')

# display the plot in Streamlit
#st.pyplot(fig)

#import streamlit as st
#from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pyplot as plt
#import numpy as np

# load iris dataset
#iris = load_iris()

# iterate over all possible pairs of features
#for i in range(iris.data.shape[1]):
#    for j in range(i + 1, iris.data.shape[1]):
#        # fit decision tree classifier
#        clf = DecisionTreeClassifier().fit(iris.data[:, [i, j]], iris.target)

        # create a meshgrid to plot the decision surface
#        x_min, x_max = iris.data[:, i].min() - 1, iris.data[:, i].max() + 1
#        y_min, y_max = iris.data[:, j].min() - 1, iris.data[:, j].max() + 1
#        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                             np.arange(y_min, y_max, 0.02))

        # predict on the meshgrid
#        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#        Z = Z.reshape(xx.shape)

        # plot the decision surface
#        fig, ax = plt.subplots()
#        ax.contourf(xx, yy, Z, alpha=0.4)
#        ax.scatter(iris.data[:, i], iris.data[:, j], c=iris.target, alpha=0.8)
#        ax.set_xlabel(iris.feature_names[i])
#        ax.set_ylabel(iris.feature_names[j])
#        ax.set_title('Decision surface of a decision tree')

        # display the plot in Streamlit
#        st.pyplot(fig)

        
#import pandas as pd
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split

# Leer el archivo csv en un dataframe llamado BD2018
#BD2018 = pd.read_csv('nombre_del_archivo.csv')

# Definir las características y el objetivo
#X = BD2018.iloc[:, 2:-2]  # Seleccionar las columnas desde la 2 hasta la antepenúltima
#y = BD2018.iloc[:, -1]    # Seleccionar la última columna como objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador k-NN con k=3
#knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador con los datos de entrenamiento
#knn.fit(X_train, y_train)

# Predecir las clases de los datos de prueba
#y_pred = knn.predict(X_test)

# Calcular la precisión del clasificador
#accuracy = knn.score(X_test, y_test)
#print('Precisión del clasificador: {:.2f}'.format(accuracy))


import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np



# load BD2018 dataset
#BD2018 = pd.read_csv('ruta/a/tu/BD2018.csv')
BD2018 = BD2018[['Nombre', 'Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]

# get feature and target columns
X = BD2018.iloc[:, 1:-2]
y = BD2018.iloc[:, -2]

# define number of columns and plots per column
num_cols = 3
plots_per_col = 5
num_plots = X.shape[1] * (X.shape[1]-1) // 2

# iterate over all possible pairs of features
plot_count = 0
for i in range(X.shape[1]):
    for j in range(i + 1, X.shape[1]):
        # fit decision tree classifier
        clf = DecisionTreeClassifier().fit(X.iloc[:, [i, j]], y)

        # create a meshgrid to plot the decision surface
        x_min, x_max = X.iloc[:, i].min() - 1, X.iloc[:, i].max() + 1
        y_min, y_max = X.iloc[:, j].min() - 1, X.iloc[:, j].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # predict on the meshgrid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # plot the decision surface
        plot_count += 1
        plt.subplot(int(np.ceil(num_plots/plots_per_col)), num_cols, plot_count)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X.iloc[:, i], X.iloc[:, j], c=y, alpha=0.8)
        plt.xlabel(X.columns[i])
        plt.ylabel(X.columns[j])

# add suptitle to the figure
plt.suptitle('Decision surfaces of a decision tree')

plt.subplots_adjust(hspace=0.8)
# display the plot in Streamlit
st.pyplot()




#import pandas as pd
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pyplot as plt
#import numpy as np
#import streamlit as st

## Cargar el conjunto de datos y asignar los nombres de las columnas
##data = pd.read_csv("BD2018.csv", header=None)
#BD2018.columns = ['Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

## Dividir el conjunto de datos en características (X) y objetivo (y)
#BD2018 = BD2018[['Edad', 'MNA', 'Fuerza', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']]

#X = BD2018.iloc[:, 0].values
#y = BD2018.iloc[:, 1].values

## Entrenar un modelo de árbol de decisión con una profundidad máxima de 3
#clf = DecisionTreeClassifier(max_depth=3)
#clf.fit(X, y)

## Crear una malla de valores para representar la superficie de decisión
#xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
#                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1))
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)

## Crear la figura y los ejes del gráfico
#fig, ax = plt.subplots()
#ax.contourf(xx, yy, Z, alpha=0.4)

## Agregar los puntos de datos al gráfico
#scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
#plt.xlabel('Edad')
#plt.ylabel('MNA')

## Agregar una leyenda al gráfico
#legend1 = ax.legend(*scatter.legend_elements(),
#                    loc="upper right", title="Target")
#ax.add_artist(legend1)

## Agregar interacción con el usuario
#tooltip = st.markdown('Hover over a point on the graph to see its values')

#def on_plot_hover(event):
#    point_x, point_y = event.xdata, event.ydata
#    idx = np.where((X[:, 0]==point_x) & (X[:, 1]==point_y))[0][0]
#    tooltip.write('Edad: {}\nMNA: {}\nFuerza: {}\nProteinas: {}\nBARTHEL: {}\nTarget: {}'.format(
#        X[idx,0], X[idx,1], X[idx,2], X[idx,3], X[idx,4], y[idx]))

#fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

# Mostrar el gráfico en Streamlit
#st.pyplot(fig)





