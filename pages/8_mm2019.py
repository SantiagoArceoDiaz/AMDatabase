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
tab1, tab2, tab3, tab4 = st.tabs(["Muestra depurada", "Estadistica descriptiva", "Clasificación de pacientes", "Análisis con teoría de conjuntos"])

with tab1:
   
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

   
        fig, ax = plt.subplots(figsize=(3,2))
        venn2019=venn2([Setdf2019, SetDBEdades], set_labels = ('Muestra de 2019', 'Muestra total'), set_colors=('red','blue'))
        st.pyplot(fig)
        st.caption("Comparativa entre los usuarios pertenecientes al año 2019 y el total, correspondiente a 2018-2021.")
    
        # Save to png
        img_fn = 'Venn 2019.png'
        fig.savefig(img_fn)

        # Prepare file for download.
        dfn = 'Venn 2019.png'
        with open(img_fn, "rb") as f:
            st.download_button(
                label="Descargar imagen",
                data=f,
                file_name=dfn,
                mime="image/png")

        st.markdown(
        """ 
        La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
        """
        )
	
	# localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Masculino."
        Hombres2019=BD2019.loc[BD2019['Sexo']=="Mas"]
        del Hombres2019['Sexo'] #Borra la columna de "Sexo", ya que es innecesaria
        Hombres2019 # Muestra el dataframe con datos de hombres.

        
        st.write(Hombres2019.describe()) # Crea un resumen estadistico sobre el dataframe "Hombres 2018".

        st.markdown(
        """ 
        La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
        """
        )

        Mujeres2019=BD2019.loc[BD2019['Sexo']=="Fem"] # localiza a todos los miembros de BD2018 que cumplen con la condicion de "Sexo" = "Femenino."
        del Mujeres2019['Sexo']
        Mujeres2019

        st.markdown(
        """ 
        La muestra se compone de 152 adultos mayores, residentes de casas de asistencia. Las pruebas se realizaron durante múltiples visitas en el año 2018. A cada uno de los pacientes que se muestran se le realizaron pruebas antropométricas, el índice de Barthel, índice mininutricional, además de pruebas sobre el contenido de proteinas en sangre. A continuación se muestra la base de datos de los participantes. 
        """
        )

        st.write(Mujeres2019.describe()) # dEscripcion del Dataframe de "Mujeres"
	
	
	
	
    with tab2:
        st.markdown(
        """ 
        # Resumen estadistico de la muestra
        Este es un resumen con la estadistica básica de la muestra. Contiene ocho filas que describen estadísticas clave para la base de datos.
        """)
        
        #carga los datos de los archivos de excel con los resultados del test de Barthel
        df2019 = pd.read_excel('2019barthel.xlsx')
        df2019 = df2019.dropna() #quita las filas que tengan NaN en algun valor
        df2019
        
        st.markdown("""
        De acuerdo al test de Cronbach, la confiabilidad del cuestionario es:
        """
                   )
        Cr=pg.cronbach_alpha(data=df2019[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete','B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])
        st.write(Cr)
       
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
        Xindepset
        
        from operator import index
        Xdepl=df2019BS.loc[df2019BS['Int_Barthel']==1.0]
        Xdeplset=set(df2019BS.loc[df2019BS['Int_Barthel']==1.0].index)
        Xdeplset
        
        from operator import index
        Xdepm=df2019BS.loc[df2019BS['Int_Barthel']==2.0]
        Xdepmset=set(df2019BS.loc[df2019BS['Int_Barthel']==2.0].index)
        Xdepmset
        
        from operator import index
        Xdeps=df2019BS.loc[df2019BS['Int_Barthel']==3.0]
        Xdepsset=set(df2019BS.loc[df2019BS['Int_Barthel']==3.0].index)
        Xdepsset
        
        from operator import index
        Xdept=df2019BS.loc[df2019BS['Int_Barthel']==4.0]
        Xdeptset=set(df2019BS.loc[df2019BS['Int_Barthel']==4.0].index)
        Xdeptset
        
        fig, ax = plt.subplots(figsize=[14, 12])
        Xindep.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xindep2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histograma de la variable independiente X")
        st.pyplot(fig)

        # Graficar histogramas de características de entrada
        #fig, axes = plt.subplots(nrows=2, ncols=4, figsize=[16, 8])
        #for i, column in enumerate(Xindep.columns):
        #        Xindep[column].plot(kind='hist', bins=20, ax=axes[i//4, i%4])
        #plt.tight_layout()

        # Guardar figura
        #plt.savefig("Xindep2019.png", bbox_inches='tight', dpi=300)

        # Graficar histogramas de características de salida
        #Xdepl_df = pd.DataFrame(Xdeplset, columns=['Xdepl'])  # convertir conjunto a DataFrame
        #Xdepl_df.hist(figsize=[14, 12], bins=20)
        #st.pyplot()
        
        
        # Mostrar imagen en Streamlit
        #st.write("Imagen de la figura guardada")
        #st.image("Xindep2019.png")
        
        fig, ax = plt.subplots(figsize=[14, 12])
        Xdepl.hist(ax=ax)
        
        # Guardar figura
        plt.savefig("Xdepl2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histograma de la variable independiente X")
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=[14, 12])
        Xdepm.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xdepm2019.png", bbox_inches='tight', dpi=300)

        # Mostrar figura en Streamlit
        st.write("Histograma de la variable independiente X")
        st.pyplot(fig)
    
        fig, ax = plt.subplots(figsize=[14, 12])
        Xdeps.hist(ax=ax)

        # Guardar figura
        plt.savefig("Xdeps2019.png", bbox_inches='tight', dpi=300)

        #Mostrar figura en Streamlit
        st.write("Histograma de la variable independiente X")
        st.pyplot(fig)
	############################################################################################################################################
	
with tab2:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """        
        )
	
with tab3:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """        
        )
	
with tab4:
   
    st.markdown(
        """ 
        # Sobre la muestra
        Se depuro la muestra para eliminar aquellos registros que presentaban informacion incompleta. En las siguientes secciones se presentan dos diferentes tipos de bases de datos: una en la que se incluyen los resultados generales de diversas pruebas y otra en la que se muestran los resultados del test de Barthel.
        """        
        )
	
	
	
	
	
	
   
