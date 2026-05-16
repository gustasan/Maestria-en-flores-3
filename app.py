
import streamlit as st
import numpy as np
import joblib
import os

# La ruta de la carpeta de modelos se asume que está en el mismo directorio que app.py
# Esto es el comportamiento estándar para despliegues en Streamlit Cloud
model_dir = os.path.dirname(__file__)

# Rutas completas a los archivos .pkl
model_path = os.path.join(model_dir, 'best_knn_model.pkl')
encoder_path = os.path.join(model_dir, 'label_encoder_species.pkl')

# Cargamos el modelo y el label encoder
try:
    with open(model_path, 'rb') as file:
        best_knn_model = joblib.load(file)
    with open(encoder_path, 'rb') as file:
        label_encoder_species = joblib.load(file)
except FileNotFoundError:
    st.error(f"""Error: No se encontraron los archivos del modelo o del codificador.
Asegúrate de que 'best_knn_model.pkl' y 'label_encoder_species.pkl' estén en el mismo directorio que app.py.""")
    st.stop() # Detiene la ejecución si los archivos no se encuentran

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Clasificador de Flores Iris", page_icon="🌸", layout="centered", initial_sidebar_state="auto")

# --- Custom CSS para una interfaz más atractiva (UX y Teoría del Color) ---
# Usamos una paleta de colores natural y suave
custom_css = """
<style>
    /* Estilos generales */
    body {
        background-color: #F8FBF8; /* Un verde muy claro, casi blanco */
        color: #333333; /* Texto oscuro para buena legibilidad */
        font-family: 'Segoe UI', sans-serif; /* Fuente moderna y legible */
    }
    .stApp {
        max-width: 900px; /* Ancho máximo para el contenido principal */
        margin: 0 auto;
        padding: 2rem;
        background-color: white; /* Contenido sobre un fondo blanco */
        box-shadow: 0 4px 8px rgba(0,0,0,0.05); /* Sombra suave para un efecto flotante */
        border-radius: 10px;
    }

    /* Títulos y encabezados */
    h1 {
        color: #4CAF50; /* Verde vibrante para el título principal */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h3 {
        color: #6CB476; /* Un verde más suave para subtítulos */
        text-align: center;
        margin-bottom: 1.5em;
    }
    h2 {
        color: #4CAF50; /* Color para secciones principales */
        border-bottom: 2px solid #E0F2E9; /* Línea suave debajo de los encabezados de sección */
        padding-bottom: 0.5em;
        margin-top: 2em;
    }

    /* Botones - Estilo de hoja */
    .stButton>button {
        background-color: #6CB476; /* Verde de botón */
        color: white;
        padding: 12px 28px;
        border: none;
        border-radius: 25px; /* Más redondeado */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Sombra mejorada */
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        margin: 20px auto; /* Centrar y dar espacio */
        display: block; /* Para que el margin auto funcione */
    }
    .stButton>button:hover {
        background-color: #4CAF50; /* Verde más oscuro al pasar el ratón */
        transform: translateY(-2px); /* Pequeño efecto de elevación */
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #E0F2E9; /* Fondo claro del slider */
        border-radius: 5px;
    }
    .stSlider > div > div > div > div > div {
        background-color: #6CB476; /* Manija del slider */
    }
    .stSlider > label {
        font-weight: bold;
        color: #333333;
    }

    /* Mensajes de resultado */
    .stAlert {
        border-radius: 8px;
        padding: 1em;
        margin-top: 1.5em;
        font-size: 1.2em;
        font-weight: bold;
    }
    .species-result {
        text-align: center;
        margin-top: 1.5em;
        padding: 1em;
        background-color: #F0FFF0; /* Un fondo muy suave para el resultado */
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
    }
    .species-result h2 {
        color: #4CAF50;
        margin-bottom: 0.5em;
        border-bottom: none;
    }
    .species-color-box {
        padding: 10px;
        border-radius: 5px;
        color: white; /* Texto blanco para asegurar contraste en los colores de especie */
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stImage {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 15px;
    }

    /* Footer */
    footer {
        text-align: center;
        margin-top: 3em;
        padding-top: 1em;
        border-top: 1px solid #EEEEEE;
        color: #AAAAAA; /* Gris claro para el footer */
        font-size: 0.9em;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# Título y descripción de la aplicación
st.title('🌸 Clasificador de Especies de Flores Iris 🌸')
st.markdown('### Una herramienta inteligente para identificar la belleza floral.')

st.write('---') # Separador visual

# Sección de entrada de datos
st.header('Ingresa las medidas de la flor')
st.markdown('Ajusta los deslizadores para las dimensiones del sépalo y del pétalo.')

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider('Largo del sépalo (cm)', 0.0, 10.0, 5.8, 0.1, help="Longitud del sépalo en centímetros.")
    petal_length = st.slider('Largo del pétalo (cm)', 0.0, 10.0, 4.3, 0.1, help="Longitud del pétalo en centímetros.")

with col2:
    sepal_width = st.slider('Ancho del sépalo (cm)', 0.0, 5.0, 3.0, 0.1, help="Ancho del sépalo en centímetros.")
    petal_width = st.slider('Ancho del pétalo (cm)', 0.0, 5.0, 1.3, 0.1, help="Ancho del pétalo en centímetros.")

# Botón para realizar la predicción
if st.button('🌿 Predecir especie de la flor'):
    st.write('---') # Separador visual antes del resultado
    # Crear el array de entrada para el modelo
    # El orden debe coincidir con el entrenamiento del modelo: [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción con el modelo KNN
    predicted_class_encoded = best_knn_model.predict(input_features)

    # Decodificar la predicción a la especie original
    predicted_species = label_encoder_species.inverse_transform(predicted_class_encoded)

    st.markdown(f"<div class='species-result'><h2>La especie predicha es: <span style='color:#4CAF50;'>{predicted_species[0].capitalize()}</span></h2></div>", unsafe_allow_html=True)

    # --- Mostrar imagen y colores basados en la especie (ejemplo) ---
    species_info = {
        'setosa': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/2/27/Iris_setosa_1.jpg',
            'color_hex': '#AEC6CF' # Azul cadete suave
        },
        'versicolor': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/2/2f/Iris_versicolor_3.jpg',
            'color_hex': '#C3A0D6' # Lavanda clara
        },
        'virginica': {
            'image': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg',
            'color_hex': '#6A5ACD' # Azul pizarra (más oscuro y rico)
        }
    }

    if predicted_species[0] in species_info:
        info = species_info[predicted_species[0]]
        st.image(info['image'], caption=f"Imagen de Iris {predicted_species[0].capitalize()}.", use_column_width=True)
        # Aseguramos que el texto del color sea blanco para contraste con colores oscuros
        st.markdown(f"<div class='species-color-box' style='background-color:{info['color_hex']};'>Color representativo: {info['color_hex']}</div>", unsafe_allow_html=True)
    else:
        st.write('No hay información visual disponible para esta especie.')

st.write('---') # Separador visual

# Footer
st.markdown("""
<footer>
    <p>Desarrollado con ❤️ y Streamlit para la predicción de especies de Iris</p>
</footer>
""", unsafe_allow_html=True)
