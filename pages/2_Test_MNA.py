import streamlit as st
import time
import numpy as np
from PIL import Image


st.set_page_config(page_title="Indice Mininutricional", page_icon="📈")

st.header("Indice Mininutricional")

st.markdown(
"""
El Índice Mini Nutritional Assessment (MNA) es una herramienta de evaluación nutricional que se utiliza para determinar el estado nutricional y la presencia de riesgo de desnutrición en pacientes adultos mayores. Fue desarrollado en Francia en la década de 1990 y se ha convertido en una herramienta de evaluación nutricional estándar en todo el mundo.

El MNA se divide en dos partes: la primera parte consiste en una entrevista con el paciente para recopilar información sobre su estado nutricional, su consumo de alimentos, su salud y su capacidad para realizar actividades cotidianas; la segunda parte incluye una evaluación física, como la medición del peso, la altura, la circunferencia del brazo y la grasa corporal.

El MNA evalúa una serie de factores, como la ingesta de alimentos, la movilidad, la presencia de enfermedades, la capacidad cognitiva y la masa muscular, entre otros. Se otorgan puntos por cada respuesta, y la puntuación total se utiliza para determinar el estado nutricional del paciente. La puntuación del MNA varía entre 0 y 30, y se clasifica en tres categorías:

- MNA ≥ 24: estado nutricional normal
- 17 ≤ MNA < 24: riesgo de desnutrición
- MNA < 17: desnutrición grave
"""
)

st.header("Indice Mininutricional en personas adultas mayores")

st.markdown(
"""
El MNA se utiliza ampliamente en la atención geriátrica para identificar a los pacientes en riesgo de desnutrición y para establecer planes de intervención nutricional para prevenir o tratar la desnutrición. Además, el MNA se puede utilizar para monitorear el estado nutricional del paciente a lo largo del tiempo y para evaluar la eficacia de las intervenciones nutricionales.
"""
)


st.write( "Video explicación sobre la evaluación mininutricional](https://www.youtube.com/watch?v=SVewQFAow2M)")

image = 'https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/MNA.png'
st.image(image, caption="Ejemplo del test de Barthel")
st.markdown(
"""
El indice Mini Nutricional (MNA) es un examen nutricional validado y una herramienta de evaluación que permite 
identificar los pacientes de edad geriátrica (mayores de 65 años) que están desnutridos o en riesgo de desnutrición. 
Fue desarrollado hace casi 20 años y es la herramienta de cribado nutricional mejor validada para las personas mayores. 
El MNA consta de 6 preguntas discriminantes (cribaje) que agilizan el proceso de selección de candidatos con posible 
desnutrición, y otras 12 preguntas adicionales que exclusivamente deberán realizarse a los individuos que han sido 
detectados como posibles casos de desnutrición por el cribaje.
"""
)

