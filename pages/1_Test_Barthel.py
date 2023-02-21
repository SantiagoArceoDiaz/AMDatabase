import streamlit as st
from PIL import Image
import time
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO


st.set_page_config(page_title="Test de Barthel", page_icon=":100:")

#image_url = "https://raw.githubusercontent.com/username/repo/master/example.png"
#image = Image.open(image_url)
#st.image(image, caption='Example image')

st.markdown("# El test de Barthel")
#image = Image.open('https://github.com/SantiagoArceoDiaz/AMDatabase/blob/main/pages/IB.jpg')
#image = Image.open('https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/IB.jpg')
#image = Image.open('https://github.com/SantiagoArceoDiaz/AMDatabase/blob/7ed451174ce84ad3ea6311783d5d6f64a430102a/pages/IB.jpg')

image = "https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/IB.jpg"
#url = "https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/IB.jpg"
#response = requests.get(url)
#image = Image.open(BytesIO(response.content))

st.image(image, caption="Ejemplo del test de Barthel")

st.markdown(
"""
Prueba sumativa en la que se evalua en desempeño del paciente en diez actividades de la vida 
cotidiana. Dichas actividades son: 

  1. **Comer:** capacidad del paciente para introducirse el alimento, masticar y tragar. La escala de puntajes es: puede introducir el alimento, masticar y tragar de forma completamente independiente (**10 puntos**), requiere ayuda para cortar o masticar los alimentos (**5 puntos**), se necesita de ayuda continua para comer (**0 puntos**).

  2. **Trasladarse de la Silla a la cama:** capacidad del paciente para moverse entre una silla y su cama sin requerir ayuda por parte de otra persona. Puede trasladarse de forma completamente independiente (**15 puntos**), require de ayuda limitada (**10 puntos**), requiere de ayuda considerable (**5 puntos**), es incapaz de trasladarse sin ayuda (**0 puntos**).

  3. **Lavarse/Aseo personal:** capacidad del paciente para mantener su higiene corporal, ducharse o limpiarse de manera autonoma. La escala es: puede realizar la actividad de forma independiente (**5 puntos**), es dependiente (**0 puntos**).

  4. **Uso del retrete:** capacidad del paciente para usar el sanitario por si mismo (involucra ir al lavavo, quitarse la ropa, hacer sus necesidades y limiparse). La escala de puntajes es: puede usar el retrete de forma independiente (**10 puntos**), requiere de ayuda parcial (**5 puntos**), es incapaz de realizar la actividad sin ayuda (**0 puntos**).
  
  5. **Bañarse ducharse:** capacidad de utilizar la regadera sin ayuda de otra persona. La escala de puntajes es: puede usar el retrete de forma independiente (**10 puntos**), requiere de ayuda parcial (**5 puntos**), es incapaz de realizar la actividad sin ayuda (**0 puntos**).

  6. **Deambular:** capacidad para caminar 50 metros sin ayuda de otra persona o andaderas. La escala de puntajes es: puede recorrer la distancia sin requerir ayuda (**15 puntos**), requiere de una cantidad minima de ayuda (**10 puntos**), puede recorrer la distancia en silla de ruedas (**5 puntos**), el sujeto es incapaz de recorrer 50 metros o permanece inmovil (**0 puntos**). 
  
  7. **Subir y bajar las escaleras:** capacidad para subir y bajar las ecaleras sin otro apoyo que el del pasamanos. La escala de puntajes es: Independiente para subir y bajar escales (**10 puntos**), necesita de ayuda fisica o verbal (**5 puntos**), es incapaz de subir y bajar escaleras (**0 puntos**). 
  
  8. **Vestirse:** capacidad para ponerse y quitarse la ropa sin requerir la ayuda de otra persona. La escala de puntajes es: La persona puede ponerse y quitarse la ropa de forma autonoma (**10 puntos**), requiere de ayuda para algunas cosas (**5 puntos**), requiere ayuda en todo momento (**0 puntos**).

  9. **Continencia o incontinencia fecal:** capacidad de contener voluntariamente los esfinteres. La escala de puntajes es: puede contener las heces (**10 puntos**), sufre episodios puntuales de incontinencia (**5 puntos**) e incontinencia habitual (**0 puntos**).

  10. **Continencia o incontinencia urinaria:** capacidad de contener voluntariamente los esfinteres. La escala de puntajes es: puede contener las heces (**10 puntos**), sufre episodios puntuales de incontinencia (**5 puntos**) e incontinencia habitual (**0 puntos**).
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
