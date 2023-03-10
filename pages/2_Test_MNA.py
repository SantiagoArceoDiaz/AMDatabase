import streamlit as st
import time
import numpy as np
from PIL import Image


st.set_page_config(page_title="Indice Mininutricional", page_icon="馃搱")

st.header("Indice Mininutricional")

st.markdown(
"""
El 脥ndice Mini Nutritional Assessment (MNA) es una herramienta de evaluaci贸n nutricional que se utiliza para determinar el estado nutricional y la presencia de riesgo de desnutrici贸n en pacientes adultos mayores. Fue desarrollado en Francia en la d茅cada de 1990 y se ha convertido en una herramienta de evaluaci贸n nutricional est谩ndar en todo el mundo.

El MNA se divide en dos partes: la primera parte consiste en una entrevista con el paciente para recopilar informaci贸n sobre su estado nutricional, su consumo de alimentos, su salud y su capacidad para realizar actividades cotidianas; la segunda parte incluye una evaluaci贸n f铆sica, como la medici贸n del peso, la altura, la circunferencia del brazo y la grasa corporal.

El MNA eval煤a una serie de factores, como la ingesta de alimentos, la movilidad, la presencia de enfermedades, la capacidad cognitiva y la masa muscular, entre otros. Se otorgan puntos por cada respuesta, y la puntuaci贸n total se utiliza para determinar el estado nutricional del paciente. La puntuaci贸n del MNA var铆a entre 0 y 30, y se clasifica en tres categor铆as:

- MNA 鈮? 24: estado nutricional normal
- 17 鈮? MNA < 24: riesgo de desnutrici贸n
- MNA < 17: desnutrici贸n grave
"""
)

st.header("Indice Mininutricional en personas adultas mayores")

st.markdown(
"""
El MNA se utiliza ampliamente en la atenci贸n geri谩trica para identificar a los pacientes en riesgo de desnutrici贸n y para establecer planes de intervenci贸n nutricional para prevenir o tratar la desnutrici贸n. Adem谩s, el MNA se puede utilizar para monitorear el estado nutricional del paciente a lo largo del tiempo y para evaluar la eficacia de las intervenciones nutricionales.
"""
)


st.write( "Video explicaci贸n sobre la evaluaci贸n mininutricional](https://www.youtube.com/watch?v=SVewQFAow2M)")

image = 'https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/MNA.png'
st.image(image, caption="Ejemplo del test de Barthel")
st.markdown(
"""
El indice Mini Nutricional (MNA) es un examen nutricional validado y una herramienta de evaluaci贸n que permite 
identificar los pacientes de edad geri谩trica (mayores de 65 a帽os) que est谩n desnutridos o en riesgo de desnutrici贸n. 
Fue desarrollado hace casi 20 a帽os y es la herramienta de cribado nutricional mejor validada para las personas mayores. 
El MNA consta de 6 preguntas discriminantes (cribaje) que agilizan el proceso de selecci贸n de candidatos con posible 
desnutrici贸n, y otras 12 preguntas adicionales que exclusivamente deber谩n realizarse a los individuos que han sido 
detectados como posibles casos de desnutrici贸n por el cribaje.
"""
)

