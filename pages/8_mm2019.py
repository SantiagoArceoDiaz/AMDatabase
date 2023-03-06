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
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import googletrans
from googletrans import Translator
translator = Translator()
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, plot_roc_curve, plot_confusion_matrix
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay




#Bloque para preparar las bases de datos

st.title("Descripción de la muestra")

st.markdown(
        """
	<div style="text-align: justify">
        La muestra de 2019 se compone de 164 adultos mayores que habitan en una casa de asistencia. A los participantes se les realizaron diferentes series de pruebas, como el test de Barthel, el índice mininutricional, la prueba de fragilidad "Share-Fi". Adicionalmente se registraron algunas cracterísticas antropométricas: fuerza de presión de brazos, circunferencia de pantorilla, velocidad de marcha. A continuación se muestran las bases de datos con los resultados de las pruebas para los pacientes registrados. La pagina se divide en 3 secciones: los resultados generales para la muestra completa y resultados particulares para las muestras de mujeres y hombres. En cada seccion se muestran: las bases de datos descargables, un resumen estadistico, la estadistica descriptiva y el analisis del test de Barthel.        
        """, unsafe_allow_html=True)        
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
df2019=pd.read_excel('2019C.xlsx')

    #del df2020['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
    #del df2020['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

df2019 = df2019.dropna() #quita las filas que tengan NaN en algun valor

df2019['Nombre']= df2019['Nombres'] + df2019['Apellidos'] #combina las columnas de nombres y apellidos en una llamada "Nombre"
del df2019['Apellidos'] # y elimina las columnas individuales.
del df2019['Nombres']
df2019['Fuerza'] = pd.to_numeric(df2019['Prom_Fuer'])

Listadf2019=df2019['Nombre'].tolist() #crea una lista a partir de los nombres de usuarios en df2018..
Setdf2019=set(Listadf2019) # convierte la lista en un conjunto (para su manejo posterior)

#st.markdown(
#        """ 
#        # Descripcion de la muestra 👋
#        La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de     los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de         proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
#        """
#        )


SetDBEdades.difference(Setdf2019) # muestra el conjunto de usuarios que aparecen en la lista de edades
    # pero no estan en la base de datos de 2018. Esto puede deberse a que no están o a que se eliminarion por tener columnas con "NaN"

ddf2019 = pd.merge(left=df2019,right=DBEdades, how="inner",on="Nombre")
    #ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
    # es decir, la intersección vista en el diagrama de Venn.

BD2019=ddf2019[['Nombre','Sexo','Edad', 'MNA', 'Marcha', 'Fuerza', 'PuntajeZ', 'Proteinas','BARTHEL', 'Int_BARTHEL']]
    #BD2018 # Cambia el orden de las columnas y se guarda como una base de datos nueva.


#tab1, tab2, tab3 = st.tabs(["Descripción de la muestra", "Estadistica básica", "Clasificación de pacientes"])
tabs1, tabs2, tabs3, tabs4 = st.tabs(["Muestra depurada", "Estadistica descriptiva", "Clasificación de pacientes", "Análisis con teoría de conjuntos"])

with tabs1:
   
    st.markdown(
        """
	<div style="text-align: justify">
        #
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Pruebas antropometricas", "Test de Barthel"])

    with tab1:
		
        st.markdown(
	"""
	<div style="text-align: justify">
	Se registro el nombre, sexo y edad de cada paciente en la base de datos. Adicionalmente, se incluyeron los resultados del indice mininutricional (MNA), velocidad de marcha y fuerza de presion de brazo, el puntaje-Z relacionado con la prueba de fragilidad share-fi, el consumo de proteinas y el resultado del test de Barthel.
	""", unsafe_allow_html=True)
	
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
        df2019=pd.read_excel('2019C.xlsx')

    #del df2020['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
    #del df2020['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

        df2019 = df2019.dropna() #quita las filas que tengan NaN en algun valor

        df2019['Nombre']= df2019['Nombres'] + df2019['Apellidos'] #combina las columnas de nombres y apellidos en una llamada "Nombre"
        del df2019['Apellidos'] # y elimina las columnas individuales.
        del df2019['Nombres']
        df2019['Fuerza'] = pd.to_numeric(df2019['Prom_Fuer'])

        Listadf2019=df2019['Nombre'].tolist() #crea una lista a partir de los nombres de usuarios en df2018..
        Setdf2019=set(Listadf2019) # convierte la lista en un conjunto (para su manejo posterior)

        st.markdown(
        """ 
        # Descripcion de la muestra 👋
        La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
        """
        )


        SetDBEdades.difference(Setdf2019) # muestra el conjunto de usuarios que aparecen en la lista de edades
        # pero no estan en la base de datos de 2018. Esto puede deberse a que no están o a que se eliminarion por tener columnas con "NaN"

        ddf2019 = pd.merge(left=df2019,right=DBEdades, how="inner",on="Nombre")
        #ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
        # es decir, la intersección vista en el diagrama de Venn.

        BD2019=ddf2019[['Nombre','Sexo','Edad', 'MNA', 'Marcha', 'Fuerza', 'PuntajeZ', 'Proteinas','BARTHEL', 'Int_BARTHEL']]
        #BD2018 # Cambia el orden de las columnas y se guarda como una base de datos nueva.

        df=BD2019

        #Seleccionar las columnas que quieres filtrar
        columnas = ['Sexo', 'Edad', 'MNA', 'Marcha', 'Fuerza', 'PuntajeZ', 'Proteinas', 'BARTHEL', 'Int_BARTHEL']

        #Crear una barra de búsqueda para cada columna en la barra lateral
        for col in columnas:
        # Verificar si la columna solo tiene valores de tipo string
            if BD2019[col].dtype == 'object':
                # Obtener los valores únicos en la columna y ordenarlos
                valores = sorted(BD2019[col].unique())
                # Crear una barra de selección para cada valor único en la columna
                seleccion = st.sidebar.multiselect(col, valores, default=valores)
                # Filtrar el dataframe en función de los valores seleccionados en la columna
                BD2019 = BD2019[BD2019[col].isin(seleccion)]
            else:
                # Obtener el rango de valores en la columna
                valores_min = BD2019[col].min()
                valores_max = BD2019[col].max()
                # Crear una barra de selección para el rango de valores en la columna
                seleccion = st.sidebar.slider(col, int(valores_min), int(valores_max), (int(valores_min), int(valores_max)))
                # Filtrar el dataframe en función de los valores seleccionados en la columna
            BD2019 = BD2019[(BD2019[col] >= seleccion[0]) & (BD2019[col] <= seleccion[1])]


	
 

        import streamlit as st
        import matplotlib.pyplot as plt 
        from matplotlib_venn import venn2

        fig, ax = plt.subplots(figsize=(3, 2))
        venn2019 = venn2([Setdf2019, SetDBEdades], set_labels=('Muestra de 2019', 'Muestra total'), set_colors=('red', 'blue'))
        fig.savefig('Venn 2019.png', dpi=300)

        with st.container():
            st.image('Venn 2019.png', width=400)
            st.caption("Comparativa entre los usuarios pertenecientes al año 2019 y el total, correspondiente a 2018-2021.")

        # Prepare file for download.
        dfn = 'Venn 2019.png'
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png")
	# Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()

	
        st.write(BD2019)
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
        La muestra de recolectada en 2018 representa compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples           visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional,           además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
        """
        )


	

        import streamlit as st
        import matplotlib.pyplot as plt

        # Cuenta el número de pacientes por sexo
        count_sexo = BD2019['Sexo'].value_counts()

        # Crea una lista con los valores de la cuenta
        values = count_sexo.values.tolist()

        # Crea una lista con las etiquetas del sexo
        labels = count_sexo.index.tolist()

        # Crea la gráfica de pastel
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(values, labels=labels)

        # Agrega el conteo de individuos de cada categoría dentro de la gráfica de pastel
        for i, v in enumerate(values):
            ax.text(i - 0.1, -0.1, str(v), fontsize=10)

        # Agrega el título a la gráfica
        ax.set_title("Distribución de pacientes por género en la muestra 2019")

        # Guarda la imagen
        fig.savefig('Pastel 2019.png', dpi=300)

        # Muestra la gráfica en Streamlit
        with st.beta_container():
            st.image('Pastel 2019.png', width=400)
            st.caption("Distribución de pacientes por género en la muestra 2019.")

        # Prepare file for download.
        dfn = 'Pastel 2019.png'
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png"
         )

	# Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()
	
	


        st.markdown(
        """ 
        A continuacion se muestra la submuestra de pacientes de genero "masculino". Si desea descargarla como "excel" o "csv" puede hacerlo presionando los botones correspondientes. 
        """
        )
	
	# localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Masculino."
        Hombres2019=BD2019.loc[BD2019['Sexo']=="Mas"]
        del Hombres2019['Sexo'] #Borra la columna de "Sexo", ya que es innecesaria
        st.write(Hombres2019) # Muestra el dataframe con datos de hombres.

        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)

        # Agregar un botón de descarga para el dataframe en la primera columna
        with col1:
            download_button(Hombres2019, 'dataframe.xlsx', 'Descargar como Excel')
            st.write('')

        # Agregar un botón de descarga para el dataframe en la segunda columna
        with col2:
            download_button_CSV(Hombres2019, 'dataframe.csv', 'Descargar como CSV')
            st.write('')
	
        st.markdown(
        """ 
        Este es el resumen estadistico de la muestra de hombres
        """
        )
        
        st.write(Hombres2019.describe()) # Crea un resumen estadistico sobre el dataframe "Hombres 2018".

        st.markdown(
        """ 
        A continuacion se muestra la submuestra de pacientes de genero "femenino". Si desea descargarla como "excel" o "csv" puede hacerlo presionando los botones correspondientes. 
        """
        )

        Mujeres2019=BD2019.loc[BD2019['Sexo']=="Fem"] # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Femenino."
        del Mujeres2019['Sexo']
        st.write(Mujeres2019)

        # Dividir la página en dos columnas
        col1, col2 = st.columns(2)

        # Agregar un botón de descarga para el dataframe en la primera columna
        with col1:
            download_button(Mujeres2019, 'dataframe.xlsx', 'Descargar como Excel')
            st.write('')

        # Agregar un botón de descarga para el dataframe en la segunda columna
        with col2:
            download_button_CSV(Mujeres2019, 'dataframe.csv', 'Descargar como CSV')
            st.write('')
	
        st.markdown(
        """ 
        Este es el resumen estadistico de la muestra de hombres
        """
        )

        st.write(Mujeres2019.describe()) # dEscripcion del Dataframe de "Mujeres"
	
	
	
	
    with tab2:
        st.markdown(
        """ 
        A continuacion se muestran los resultados del test de Barthel para los pacientes de la muestra de 2019.
        """)
        
        #carga los datos de los archivos de excel con los resultados del test de Barthel
        df2019 = pd.read_excel('2019barthel.xlsx')
        df2019 = df2019.dropna() #quita las filas que tengan NaN en algun valor
        df2019
        
        df2019Hombres=df2019.loc[df2019['Sexo']==2.0]
        df2019Mujeres=df2019.loc[df2019['Sexo']==1.0]
	
        st.markdown("""
        El test de Cronbach permite evaluar la confiabilidad de las respuestas de los pacientes al cuestionario. De acuerdo al test de Cronbach, la confiabilidad del cuestionario es:
        """
                   )
        Cr=pg.cronbach_alpha(data=df2019[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete','B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])
        st.write("Nivel de confiabilidad", Cr)
       
        st.markdown("""
        En el caso de las submuestras de hombres y mujeres, el test de Cronbach da resultados marcadamente distintos.
        """
                   )
        CrH=pg.cronbach_alpha(data=df2019Hombres[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete','B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])
        st.write("Nivel de confiabilidad para las respuestas de los pacientes de genero masculino", CrH)
        CrM=pg.cronbach_alpha(data=df2019Mujeres[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete','B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])
        st.write("Nivel de confiabilidad para las respuestas de los pacientes de genero masculino", CrM)



        st.markdown(
        """ 
        # Resumen estadistico de la muestra
        Este es un resumen con la estadistica básica de la muestra. Contiene ocho filas que describen estadísticas clave para la base de datos.
        """)

        ListadfBarth2019=df2019['Nombre'].tolist() # crea una lista con los usuarios de 2019
        SetdfBarth2019=set(Listadf2019) # crea un conjunto a partir de la lista de usuarios de 2019   
        
        df2019BS=df2019[['Nombre','Sexo','Edad','B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
        'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina',
        'Int_Barthel']]
        
        from operator import index
        Xindep=df2019BS.loc[df2019BS['Int_Barthel']==0.0]
        Xindepset=set(df2019BS.loc[df2019BS['Int_Barthel']==0.0].index)
        st.write("Los pacientes con un diagnostico de dependencia nula son:", Xindepset)
        
        from operator import index
        Xdepl=df2019BS.loc[df2019BS['Int_Barthel']==1.0]
        Xdeplset=set(df2019BS.loc[df2019BS['Int_Barthel']==1.0].index)
        st.write("Los pacientes con un diagnostico de dependencia leve son:", Xdeplset)
        
        from operator import index
        Xdepm=df2019BS.loc[df2019BS['Int_Barthel']==2.0]
        Xdepmset=set(df2019BS.loc[df2019BS['Int_Barthel']==2.0].index)
        st.write("Los pacientes con un diagnostico de dependencia moderada son:", Xdepmset)
        
        from operator import index
        Xdeps=df2019BS.loc[df2019BS['Int_Barthel']==3.0]
        Xdepsset=set(df2019BS.loc[df2019BS['Int_Barthel']==3.0].index)
        st.write("Los pacientes con un diagnostico de dependencia severa son:", Xdepsset)
        
        from operator import index
        Xdept=df2019BS.loc[df2019BS['Int_Barthel']==4.0]
        Xdeptset=set(df2019BS.loc[df2019BS['Int_Barthel']==4.0].index)
        st.write("Los pacientes con un diagnostico de dependencia total son:", Xdeptset)
        
        Independientes, ax = plt.subplots(figsize=[14, 12])
        Xindep.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xindep2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histogramas de los puntajes obtenidos en cada pregunta para pacientes diagnosticados como independientes")
        st.pyplot(Independientes)

        # Prepare file for download.
        dfn = "Xindep2019.png"
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png"
         )
               
        # Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()
	
	############
        Dependientesleves, ax = plt.subplots(figsize=[14, 12])
        Xdepl.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xdepl2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histogramas de los puntajes obtenidos en cada pregunta para pacientes diagnosticados con dependencia leve")
        st.pyplot(Dependientesleves)
 
	
	# Prepare file for download.
        dfn = "Xdepl2019.png"
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png"
         )
	
	# Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()

        Dependientesmoderado, ax = plt.subplots(figsize=[14, 12])
        Xdepm.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xdepm2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histogramas de los puntajes obtenidos en cada pregunta para pacientes diagnosticados con dependencia moderada")
        st.pyplot(Dependientesmoderado)

        # Prepare file for download.
        dfn = "Xdepm2019.png"
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png"
         )
	
	# Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()

        Dependientesseveros, ax = plt.subplots(figsize=[14, 12])
        Xdeps.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xdeps2019.png", bbox_inches='tight', dpi=300)

        #Mostrar figura en Streamlit
        st.write("Histogramas de los puntajes obtenidos en cada pregunta para pacientes diagnosticados con dependencia severa")
        st.pyplot(Dependientesseveros)

	
	# Prepare file for download.
        dfn = "Xdeps2019.png"
        with open(dfn, "rb") as f:
            st.download_button(
            label="Descargar imagen",
            data=f,
            file_name=dfn,
            mime="image/png"
         )

	# Limpiar la figura para evitar advertencias en la siguiente subtrama
        ax.cla()
	
	############################################################################################################################################
	
with tabs2:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """        
        )
    Barras2019, axes = plt.subplots(3, 2, figsize=(10, 10))
    sns.histplot(BD2019['Edad'], ax=axes[0,0], kde=True, line_kws={'linewidth': 2})
    sns.histplot(BD2019['MNA'], ax=axes[0,1], kde=True, line_kws={'linewidth': 2})
    sns.histplot(BD2019['Marcha'], ax=axes[1,0], kde=True, line_kws={'linewidth': 2})
    sns.histplot(BD2019['Fuerza'], ax=axes[1,1], kde=True, line_kws={'linewidth': 2})                      
    sns.histplot(BD2019['PuntajeZ'], ax=axes[2,0], kde=True, line_kws={'linewidth': 2})
    sns.histplot(BD2019['Proteinas'], ax=axes[2,1], kde=True, line_kws={'linewidth': 2})                      
    st.pyplot(Barras2019)

    st.markdown(
    """
    La grafica muestra los histogramas de la distribucion de frecuencias de los paramtero relevantes para la base de datos: Edad [años], Índice Mininutricional [puntaje], Fuerza promedio de antebrazo [kilogramos] y consumo diario de proteinas [gramos]. La línea azul representa una estimación de la densidad de probabilidad de la variable (kde es el acrónimo de "Kernel Density Estimate"). En los histogramas se muestra la distribución de frecuencia de los valores de cada variable. En el eje x se encuentran los valores de la variable y en el eje y se encuentra la frecuencia de los valores.
    """
    )

#######################01#################################



    chart1=altcat.catplot(BD2019,
               height=250,
               width=350,
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
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

    chart2=altcat.catplot(BD2019,
               height=250,
               width=350,
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
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

    chart3=altcat.catplot(BD2019,
               height=250,
               width=350,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('PuntajeZ:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip("Fuerza"),
                             alt.Tooltip("MNA"),
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

    chart4=altcat.catplot(BD2019,
               height=250,
               width=350,
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
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

    chart5=altcat.catplot(BD2019,
               height=250,
               width=350,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Marcha:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip("Fuerza"),
                             alt.Tooltip("MNA"),
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()

    chart6=altcat.catplot(BD2019,
               height=250,
               width=350,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('BARTHEL:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip('Fuerza'),
                             alt.Tooltip("MNA"),
                             alt.Tooltip("Marcha"),
                             alt.Tooltip("PuntajeZ"),
                             alt.Tooltip("BARTHEL"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


    cajas2019=alt.vconcat(alt.hconcat(chart1, chart2),alt.hconcat(chart3, chart4),alt.hconcat(chart5, chart6))


    st.altair_chart(cajas2019)


    selection = alt.selection_multi(fields=['Sexo'], bind='legend')
    chart1 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='Fuerza',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip('Fuerza'),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()

    chart2 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='Proteinas',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip('Fuerza'),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()

    chart3 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='MNA',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip('Fuerza'),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()

    chart4 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='Marcha',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip("MNA"),
         alt.Tooltip("Fuerza"),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()

    chart5 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='PuntajeZ',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip("Fuerza"),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()

    chart6 = alt.Chart(BD2019).mark_circle(size=50).encode(
         x='Edad', y='BARTHEL',
         color='Sexo',
         tooltip=[alt.Tooltip("Nombre"),
         alt.Tooltip("MNA"),
         alt.Tooltip('Fuerza'),
         alt.Tooltip("Marcha"),
         alt.Tooltip("PuntajeZ"),
         alt.Tooltip("BARTHEL"),
         ],
         opacity=alt.condition(selection, alt.value(1), alt.value(0))
         ).properties(
         height=300, width=350
         ).add_selection(
         selection
         ).interactive()


    correlaciones2019=alt.vconcat(alt.hconcat(chart1, chart2),alt.hconcat(chart3, chart6))

    st.altair_chart(correlaciones2019)



    st.markdown(
    """ 
    Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

    # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Masculino."
    Hombres2019=BD2019.loc[BD2019['Sexo']=="Mas"]
    del Hombres2019['Sexo'] #Borra la columna de "Sexo", ya que es innecesaria
    Hombres2019 # Muestra el dataframe con datos de hombres.

    st.markdown(
    """ 
    Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

    Hombres2019.describe() # Crea un resumen estadistico sobre el dataframe "Hombres 2018".

    st.markdown(
    """ 
    # Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

    Mujeres2019=BD2019.loc[BD2019['Sexo']=="Fem"] # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Femenino."

    del Mujeres2019['Sexo']
    Mujeres2019

    st.markdown(
    """ 
    Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )

    Mujeres2019.describe() # dEscripcion del Dataframe de "Mujeres"




    Hombres201960=BD2019.loc[((BD2019['Edad'] <= 60) & (BD2019['Sexo']=='Mas'))]
    del Hombres201960['Sexo']
    Hombres201970=BD2019.loc[((BD2019['Edad'] > 60) & (BD2019['Edad'] <= 70) & (BD2019['Sexo'] == 'Mas'))]
    del Hombres201970['Sexo']
    Hombres201980=BD2019.loc[((BD2019['Edad'] > 70) & (BD2019['Edad'] <= 80) & (BD2019['Sexo'] == 'Mas'))]
    del Hombres201980['Sexo']
    Hombres201990=BD2019.loc[((BD2019['Edad'] > 80) & (BD2019['Edad'] <= 90) & (BD2019['Sexo'] == 'Mas'))]
    del Hombres201990['Sexo']
    Hombres2019100=BD2019.loc[((BD2019['Edad'] > 90) & (BD2019['Sexo'] == 'Mas'))]
    del Hombres2019100['Sexo']


    Mujeres201960=BD2019.loc[((BD2019['Edad']<=60) & (BD2019['Sexo']=='Fem'))]
    del Mujeres201960['Sexo']
    Mujeres201970=BD2019.loc[((BD2019['Edad'] >60) & (BD2019['Edad']<=70) & (BD2019['Sexo']=='Fem'))]
    del Mujeres201970['Sexo']
    Mujeres201980=BD2019.loc[((BD2019['Edad'] >70) & (BD2019['Edad']<=80) & (BD2019['Sexo']=='Fem'))]
    del Mujeres201980['Sexo']
    Mujeres201990=BD2019.loc[((BD2019['Edad'] >80) & (BD2019['Edad']<=90) & (BD2019['Sexo']=='Fem'))]
    del Mujeres201990['Sexo']
    Mujeres2019100=BD2019.loc[((BD2019['Edad'] >90) & (BD2019['Sexo']=='Fem'))]
    del Mujeres2019100['Sexo']





    st.markdown(
    """ 
    Descripcion de la muestra 👋
    La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
    """
    )




    #import seaborn as sns
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    # Here we create a figure instance, and two subplots
    CalorHombres2019 = plt.figure(figsize = (20,20)) # width x height
    ax1 = CalorHombres2019.add_subplot(2, 2, 1) # row, column, position
    ax2 = CalorHombres2019.add_subplot(2, 2, 2)
    ax3 = CalorHombres2019.add_subplot(2, 2, 3)


    # We use ax parameter to tell seaborn which subplot to use for this plot
    sns.heatmap(data=Hombres201970.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax1, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    sns.heatmap(data=Hombres201980.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax2, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    sns.heatmap(data=Hombres201990.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax3, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    st.pyplot(CalorHombres2019)





    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    # Here we create a figure instance, and two subplots
    CalorMujeres2019 = plt.figure(figsize = (20,20)) # width x height
    ax1 = CalorMujeres2019.add_subplot(2, 2, 1) # row, column, position
    ax2 = CalorMujeres2019.add_subplot(2, 2, 2)
    ax3 = CalorMujeres2019.add_subplot(2, 2, 3)
    ax4 = CalorMujeres2019.add_subplot(2, 2, 4)


    # We use ax parameter to tell seaborn which subplot to use for this plot
    sns.heatmap(data=Mujeres201960.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax1, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    sns.heatmap(data=Mujeres201970.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax2, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    sns.heatmap(data=Mujeres201980.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax3, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    sns.heatmap(data=Mujeres201990.corr().loc[:'BARTHEL', :"BARTHEL"], ax=ax4, cmap = cmap, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 12})
    st.pyplot(CalorMujeres2019)	

	
	#############################################################################################################################################
	
with tabs3:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
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


    X = BD2019.iloc[:,2:-2].values
    y = BD2019.iloc[:,-2].values

    sc=StandardScaler()
    X_transform=sc.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_transform, y)
    lin_reg.intercept_, lin_reg.coef_

    # Find the optimal theta values using the custom function
    theta_optimal = Multivariable_Linear_Regression(X_transform, y, 0.03, 30000)

    # Create a new dataframe with the original data and predicted values
    X_transform_df = pd.DataFrame(X_transform, columns=['Edad', 'MNA', 'Marcha', 'Fuerza', 'PuntajeZ', 'Proteinas'])
    predictions = np.dot(X_transform_df, theta_optimal[1:]) + theta_optimal[0]
    BD2019_with_predictions = BD2019.assign(Predicted_BARTHEL=predictions)

    # Print the new dataframe with predictions
    st.write(BD2019_with_predictions)

    from sklearn.datasets import load_iris
    from sklearn import tree
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn import tree



    X = BD2019.iloc[:,2:-2].values
    y = BD2019.iloc[:,-1].values


    # Definir el algoritmo de clasificación y ajustar los datos
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)



    # Crear los deslizadores
    Edad = st.slider("Edad", 60, 100, 75)
    MNA = st.slider("MNA", 0, 30, 15)
    Fuerza = st.slider("Fuerza", 0, 150, 75)
    Proteinas = st.slider("Proteinas", 0, 200, 100)
    PuntajeZ = st.slider("PuntajeZ", 0, 200, 100)
    Marcha = st.slider("Marcha", 0, 200, 100)



    # Clasificar el objeto
    prediction = clf.predict([[Edad, MNA, Fuerza, Proteinas, PuntajeZ, Marcha]])
    print("El objeto pertenece a la clase:", prediction[0])



    train_data = BD2019.iloc[:-20,2:-2].values

    train_target = BD2019.iloc[:-20, -1].values

    test_data = BD2019.iloc[-20:].values


    test_target = BD2019.iloc[-20:, -1].values

    # Train a decision tree classifier
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    # Use the trained classifier to classify a new object
    new_object = np.array([[Edad, MNA, Fuerza, Proteinas, PuntajeZ, Marcha]])
    prediction = clf.predict(new_object)



    clf = DecisionTreeClassifier()
    clf.fit(X, y)


    class_names=BD2019.columns[-1] #
    tree_rules = sk.tree.export_text(clf, feature_names=BD2019.columns[2:-2].tolist())

    st.text(tree_rules)



    tree.plot_tree(clf, filled=True, feature_names=BD2019.columns[2:-2].tolist(), class_names=BD2019.columns[-1])
    plt.show()
    st.pyplot()

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


    class_names = ['0', '1', '2']
    rules = get_rules(clf, BD2019.columns[2:-2].tolist(), BD2019.columns[-1])
    for r in rules:
        st.write(r)

#######################################

    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt
    import numpy as np



    BD2019 = BD2019[['Nombre','Edad', 'Marcha', 'MNA', 'Fuerza', 'Proteinas', 'PuntajeZ', 'BARTHEL', 'Int_BARTHEL']]
## get feature and target columns
    X = BD2019.iloc[:, 1:-2]
    y = BD2019.iloc[:, -2]

# Modificamos el número de filas y columnas
    num_cols = 3
    plots_per_col = 5
    num_plots = X.shape[1] * (X.shape[1]-1) // 2

# Eliminamos algunos subgráficos si es necesario para que el número total de subgráficos sea un múltiplo de 3
    num_extra_plots = num_plots % num_cols
    if num_extra_plots > 0:
        num_plots -= num_extra_plots

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
            if plot_count <= num_plots:
                plt.subplot(int(num_plots/num_cols), num_cols, plot_count)
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X.iloc[:, i], X.iloc[:, j], c=y, alpha=0.8)
                plt.xlabel(X.columns[i])
                plt.ylabel(X.columns[j])
                #plt.remove()

    # add suptitle to the figure
    plt.suptitle('Decision surfaces of a decision tree')
    plt.subplots_adjust(hspace=0.8)
    # display the plot in Streamlit
    st.pyplot()

###################################

    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt
    import numpy as np

    BD2019 = BD2019[['Nombre','Edad', 'Marcha', 'MNA', 'Fuerza', 'Proteinas', 'PuntajeZ', 'BARTHEL', 'Int_BARTHEL']]

    ## get feature and target columns
    X = BD2019.iloc[:, 1:-2]
    y = BD2019.iloc[:, -2]

    # Modificamos el número de filas y columnas
    num_cols = 3
    plots_per_col = 5
    num_plots = X.shape[1] * (X.shape[1]-1) // 2

    # Eliminamos algunos subgráficos si es necesario para que el número total de subgráficos sea un múltiplo de 3
    num_extra_plots = num_plots % num_cols
    if num_extra_plots > 0:
        num_plots -= num_extra_plots

    # iterate over all possible pairs of features
    plot_count = 0
    fig, axs = plt.subplots(int(num_plots/num_cols), num_cols, figsize=(12,12))
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
            if plot_count <= num_plots:
                row = (plot_count-1) // num_cols
                col = (plot_count-1) % num_cols
                axs[row][col].contourf(xx, yy, Z, alpha=0.4)
                axs[row][col].scatter(X.iloc[:, i], X.iloc[:, j], c=y, alpha=0.8)
                axs[row][col].set_xlabel(X.columns[i])
                axs[row][col].set_ylabel(X.columns[j])

    # add suptitle to the figure
    fig.suptitle('Decision surfaces of a decision tree')
    fig.subplots_adjust(hspace=0.8)
    # display the plot in Streamlit
    st.pyplot(fig)



    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    # define a function to plot the decision surface
    def plot_decision_surface(X, y, feature1, feature2):
        clf = DecisionTreeClassifier().fit(X.loc[:, [feature1, feature2]], y)
        x_min, x_max = X.loc[:, feature1].min() - 1, X.loc[:, feature1].max() + 1
        y_min, y_max = X.loc[:, feature2].min() - 1, X.loc[:, feature2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X.loc[:, feature1], X.loc[:, feature2], c=y, alpha=0.8)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('Decision surface of a decision tree')

    # load the data
    BD2019 = BD2019[['Nombre','Edad', 'Marcha', 'MNA', 'Fuerza', 'Proteinas', 'PuntajeZ', 'BARTHEL', 'Int_BARTHEL']]

    # get feature and target columns
    X = BD2019.iloc[:, 1:-2]
    y = BD2019.iloc[:, -2]

    # set up the sidebar inputs
    st.sidebar.header('Select two features to display the decision surface')
    feature1 = st.sidebar.selectbox('First feature', X.columns)
    feature2 = st.sidebar.selectbox('Second feature', X.columns)

    # plot the decision surface based on the selected features
    plot_decision_surface(X, y, feature1, feature2)
    # display the plot in Streamlit
    plt.show()
    st.pyplot()



    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score


    # Separar el conjunto de entrenamiento y de prueba
 
    BD2019 = BD2019[['Nombre','Edad', 'Marcha', 'MNA', 'Fuerza', 'Proteinas', 'PuntajeZ', 'BARTHEL', 'Int_BARTHEL']]
    ## get feature and target columns
    X = BD2019.iloc[:, 1:-2]
    y = BD2019.iloc[:, -1]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Crear un clasificador de random forest
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # Entrenar el clasificador con los datos de entrenamiento
    classifier.fit(X_train, y_train)

    # Predecir las clases del conjunto de prueba
    y_pred = classifier.predict(X_test)
    st.write("valores predichos", y_pred)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    #print("Precisión:", accuracy)
    st.write("## Resultados de Random Forest")
    st.write("Precisión:", accuracy)
    
    # Graficar importancia de características
    feature_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)
    feature_importances.plot(kind='barh')
    plt.title("Importancia de características")
    st.pyplot()
 
        
    
    # Graficar árbol
    plt.figure(figsize=(15,10))
    tree.plot_tree(classifier.estimators_[0], feature_names=X_train.columns, filled=True)
    plt.title("Árbol de decisión")
    st.pyplot()

    
    # Graficar matriz de confusión
    cf=confusion_matrix(y_test, y_pred)
    st.write("Matriz de confusión",cf)

    import seaborn as sns
    import matplotlib.pyplot as plt


    ax = sns.heatmap(cf/np.sum(cf), annot=True, 
            fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Flower Category')
    ax.set_ylabel('Actual Flower Category ');

## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
    ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])

## Display the visualization of the Confusion Matrix.
    st.pyplot()

    # Generar matriz de confusión
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

# Crear gráfica de errores de predicción
    plt.title("Matriz de confusión")
    plt.ylabel('Valores reales')
    plt.xlabel('Valores predichos')
    tick_marks = np.arange(len(set(y))) + 0.5
    plt.xticks(tick_marks, set(y))
    plt.yticks(tick_marks, set(y))
    plt.gca().set_xticklabels(sorted(set(y)))
    plt.gca().set_yticklabels(sorted(set(y)))
    plt.gca().xaxis.tick_top()
    threshold = cf_matrix.max() / 2
    #for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
                plt.text(j, i, format(cf_matrix[i, j], '.2f'),
                horizontalalignment="center",
                color="white" if cf_matrix[i, j] > threshold else "black")
    plt.axhline(y=0.5, xmin=0, xmax=3, color='black', linewidth=2)
    plt.axvline(x=0.5, ymin=0, ymax=3, color='black', linewidth=2)
    plt.show()
    st.pyplot()

	
	
	
with tabs4:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """        
        )
	
	
	
	
	
	
   
