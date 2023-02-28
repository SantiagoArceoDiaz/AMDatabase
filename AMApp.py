import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Sobre la Sarcopenia 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Se define la sarcopenia como la perdida progresiva de masa muscular, asociada con la edad. Esta condicion es tipica, aunque
    no exclusiva, en adultos mayores y puede generar impactos negativos en la fuerza y la habilidad funcional al realizar 
    tareas propias de la vida cotidiana. 

    Los adultos registran una perdida promedio de masa muscular de entre 3 y 8 porciento a patir de los 30 años. Algunso factores 
    que se han asociado con la aparicion de la sarcopenia son: la inactividad fisica, la edad y una dieta deficiente. 

    En el caso de los adultos mayores, la ausencia de ejercicio tiene multiples consecuencias asociadas con la perdida de masa
    muscular: un riesgo incrementado de caidas y fracturas, un aumento en la sensacion de fatiga durante el dia y una disminucion 
    en la resistencia muscular. Entre las habilidades necesarias para la vida cotidiana que se ven afectadas por la sarcopenia 
    estan: la velocidad de marcha, caídas, incapacidad para subir escaleras y en general, debilidad en las extremidades 
    inferiores. La ingesta insuficiente de proteinas en la dieta esta relacionada con la perdida de masa muscular en hombres y 
    mujeres de edades comprendidas entre 70 y 79 años.

    La base de datos antropometricos esta conformada por los resultados de pruebas realizadas en adultos mayores que 
    residen en casas de asistencia. Estas pruebas incluyen: los test de Barthel e indice mininutricional, mediciones antropometricas de
    fuerza de presion y velocidad de marcha, la ingesta diaria de proteinas y la estimacion del riesgo de fragilidad (mediante el Z-score)

"""
)





#st.title("Language Translator :smile:")

#import streamlit as st
#from textblob import TextBlob

#def translate(text, dest_lang):
#    blob = TextBlob(text)
#    translated = blob.translate(to=dest_lang)
#    return ' '.join(map(str, translated))

# Create a text area for user input
#text_input = st.text_area('Enter text to translate:', height=200)

# Create a selectbox for destination language
#dest_lang = st.selectbox('Select destination language:', ['en', 'fr', 'es'])

# Create a button to translate the text
#if st.button('Translate'):
#    translation = translate(text_input, dest_lang)
#    st.write('Translated text:')
#    st.write(translation)



#import streamlit as st
#from textblob import TextBlob
#from streamlit.components.v1 import html
#from streamlit.report_thread import add_report_ctx


# Define a function to translate text
#def translate(text, dest_lang):
#    blob = TextBlob(text)
#    translated = blob.translate(to=dest_lang)
#    return str(translated)

# Create a selectbox for destination language
#dest_lang = st.selectbox('Select destination language:', ['en', 'fr', 'es'])

# Create a button to translate the page
#if st.button('Translate'):
#    # Get the page content as HTML
#    page = st._add_report_ctx().get_report_adhoc()
#    html_content = page.get_html_content()

#    # Translate the HTML content
#    translated_html = translate(html_content, dest_lang)

    # Show the translated HTML content
#    html(translated_html, scrolling=True, width=800, height=1000)



tab1, tab2, tab3 = st.tabs(["Fuerza de brazo", "Circunferencia de Pantorrilla", "Velocidad de Marcha"])

with tab1:
    st.header("Fuerza de brazo")
    st.markdown(
        """
        Se mide la fuerza de presion del paciente utilizando un dinamometro. La prueba se realiza 
        en el brazo dominante, flexionado a 90 grados, con el paciente sentado en una silla. La unidad de medida son los kilogramos. 
        Esta prueba permite establecer puntos de corte para fragilidad, dependiendo del indice de masa corporal del paciente. 
        En Hombres estos son menos de 29 kg (IMC<24), 30 kg (IMC[24,28]) y 32 kg (IMC > 28). En la mujeres los puntos de corte son: 
        menor a 17 kg (IMC < 23), menor a 17.3 kg (IMC [23.1,26]), menor a 18 kg (IMC [26,29]) y menor a 21 kg (IMC > 29).
        """
    )

    
    image="https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/547b9d1f27ac2e5c5396f35cc85507e74b92404f/pages/Dina.png"
    st.image(image, caption="Medicion de la fuerza de presion usando un dinamometro digital")


with tab2:
    st.header("Circunferencia de Pantorrilla")
    st.markdown(
    """La prueba de circunferencia de pantorrilla es una medida clínica utilizada para evaluar la masa muscular periférica en adultos mayores. Esta prueba es una evaluación simple y no invasiva de la masa muscular que se puede realizar en un entorno clínico o en el hogar. La circunferencia de la pantorrilla es un indicador útil de la masa muscular periférica debido a la alta correlación entre la circunferencia de la pantorrilla y la masa muscular total. La disminución de la masa muscular periférica es un indicador común de la disminución de la fuerza y ​​la función muscular en adultos mayores, lo que se asocia con una mayor discapacidad, caídas y mortalidad. Para realizar la prueba de circunferencia de pantorrilla, se mide la circunferencia de la pantorrilla desnuda en la pierna dominante, en un punto específico, generalmente en la parte más ancha de la pantorrilla. La medida se toma utilizando una cinta métrica flexible y se registra en centímetros. Los valores normales de la circunferencia de la pantorrilla pueden variar según la edad, el sexo y la etnia, pero generalmente se considera normal una medida superior a 31 cm en mujeres y 34 cm en hombres. La prueba de circunferencia de pantorrilla es una herramienta útil para la evaluación de la masa muscular periférica en adultos mayores, pero debe usarse junto con otras medidas clínicas y pruebas de función muscular para una evaluación más completa del estado de la masa muscular y la fuerza en los adultos mayores.
    """
    )
    image = 'https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/Calf.jpeg'
    st.image(image, caption="Prueba de velocidad de marcha de 4 metros")



with tab3:
    st.header("Velocidad de Marcha")
    st.markdown(
    """ La prueba de velocidad de marcha, también conocida como prueba de la marcha de 4 metros, es una evaluación simple y rápida que se utiliza comúnmente en adultos mayores para medir su velocidad de marcha y su capacidad funcional. La prueba implica cronometrar el tiempo que tarda una persona en caminar cuatro metros a su ritmo habitual. La velocidad de marcha se considera un predictor importante de la capacidad funcional de los adultos mayores, lo que significa que puede ser un indicador de su capacidad para realizar actividades diarias y su calidad de vida en general. En particular, se ha demostrado que la velocidad de marcha se correlaciona con la capacidad para realizar actividades básicas de la vida diaria, como levantarse de una silla, caminar y subir escaleras, así como con la capacidad para realizar actividades instrumentales de la vida diaria, como hacer compras, cocinar y manejar el dinero. Además de ser una herramienta útil para la evaluación de la capacidad funcional, la prueba de velocidad de marcha también se ha utilizado como predictor de la mortalidad en adultos mayores. Se ha demostrado que una velocidad de marcha lenta se asocia con un mayor riesgo de mortalidad en esta población. En resumen, la prueba de velocidad de marcha es una herramienta importante para la evaluación de la capacidad funcional y la salud en adultos mayores. Permite a los profesionales de la salud identificar a las personas que pueden estar en mayor riesgo de limitaciones funcionales y desarrollar planes de tratamiento y prevención para mejorar su calidad de vida y reducir su riesgo de mortalidad.
    """
    )
    image = 'https://raw.githubusercontent.com/SantiagoArceoDiaz/AMDatabase/main/pages/VelMarch.PNG'
    st.image(image, caption="Prueba de velocidad de marcha de 4 metros")
    st.write( "check out thisPrueba de velocidad de marcha](https://www.youtube.com/watch?v=zJZGW-TD23E)")

    
    st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # Bibliografia
    - http://scielo.isciii.es/pdf/nh/v21s3/art06.pdf
    - http://www.redalyc.org/pdf/3092/309226783003.pdf

    """    
    )
