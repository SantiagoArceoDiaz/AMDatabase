import streamlit as st
import time
import numpy as np
from PIL import Image


st.set_page_config(page_title="Indice Mininutricional", page_icon="📈")

st.markdown("# Indice Mininutricional")
image = Image.open('https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/MNA.png')
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
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
