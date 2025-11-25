import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np


# --- Configuración de la Aplicación ---
st.set_page_config(
    page_title="Modelo Simplificado de Intención de Compra 🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️🛒 Modelo de Predicción de Intención de Compra (E-commerce)")
st.markdown("Ajusta las **variables clave** para predecir la intención de compra. Ahora con manejo robusto de tipos de datos para evitar errores.")

# --- Lista Maestra de Características y Valores por Defecto ---
# El orden CORRECTO de las 13 columnas que el modelo espera.
FULL_FEATURE_LIST = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay', 'Browser', 'VisitorType', 'Weekend'
] 

# Valores por defecto para las variables NO expuestas.
DEFAULT_VALUES = {
    # CONTEOS (Se forzará a INT en el DataFrame)
    'Administrative': 0,
    'Informational': 0,
    'ProductRelated': 18,

    # TASAS Y DÍAS ESPECIALES (Se forzará a FLOAT)
    'BounceRates': 0.02,
    'ExitRates': 0.04,
    'SpecialDay': 0.0,
    
    # CATEGÓRICAS ORIGINALES (Se forzará a INT)
    'Browser': 2,
    # 'VisitorType' se hace interactivo
}

# Mapping de VisitorType: Asegúrate de que estos valores (0, 1, 2) coincidan con tu Label Encoding.
VISITOR_MAPPING = {
    "Visitante Recurrente": 2, # El más común
    "Nuevo Visitante": 1, 
} # La opción 'Otro': 0 se omite de la interfaz.


# --- Carga del Modelo ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_shoppers.pkl')
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo. Asegúrate de que 'best_model_shoppers.pkl' esté en la carpeta 'models'. Error: {e}")
        return None

model = load_model()

if model is not None:
    
    st.header("Parámetros de la Sesión")

    # --- Interfaz de Usuario (Inputs de Duración y Valor) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Tiempo y Valor ⏳")
        # 1. PageValues
        page_values = st.slider(
            "PageValues (Valor Promedio)", 0.0, 400.0, 0.0, 0.1
        )
        
        # 2. ProductRelated_Duration
        product_duration = st.slider(
            "ProductRelated_Duration (Duración en Prod.)", 0.0, 10000.0, 600.0, 1.0
        )

    with col2:
        st.subheader("Duración en Servicios ⚙️")
        # 3. Administrative_Duration
        admin_duration = st.slider(
            "Administrative_Duration (Duración en Admin.)", 0.0, 1000.0, 7.5, 0.1
        )
        
        # 4. Informational_Duration
        info_duration = st.slider(
            "Informational_Duration (Duración en Info.)", 0.0, 500.0, 0.0, 0.1
        )

    with col3:
        st.subheader("Contexto del Usuario 🗓️")
        
        # 5. Weekend (Checkbox Solicitado)
        weekend = st.checkbox("¿La sesión ocurrió durante el **Fin de Semana**?", value=False)
        weekend_value = 1 if weekend else 0
        
        # 6. VisitorType (Radio Buttons para selección)
        visitor_type_selection = st.radio(
            "Tipo de Visitante:",
            list(VISITOR_MAPPING.keys()),
            index=0, # Por defecto, 'Visitante Recurrente' (el valor 2 del mapping)
        )
        # Convertir la selección de texto a su valor numérico
        visitor_type_value = VISITOR_MAPPING[visitor_type_selection]
        
        st.info("El resto de variables (conteos, tasas y navegador) están fijadas a sus valores promedio o más comunes.")

    
    # --- Botón de Predicción ---
    st.markdown("---")
    if st.button("🚀🛒 Predecir Intención de Compra"):
        
        # 1. Crear el diccionario de entrada combinando valores fijos e interactivos
        input_data_dict = {
            # Variables de conteo/ID (se fuerzan a INT)
            'Administrative': int(DEFAULT_VALUES['Administrative']),
            'Informational': int(DEFAULT_VALUES['Informational']),
            'ProductRelated': int(DEFAULT_VALUES['ProductRelated']),
            'Browser': int(DEFAULT_VALUES['Browser']),
            'VisitorType': int(visitor_type_value),
            'Weekend': int(weekend_value),
            
            # Variables de Duración/Tasa/Valor (se fuerzan a FLOAT)
            'Administrative_Duration': float(admin_duration),
            'Informational_Duration': float(info_duration),
            'ProductRelated_Duration': float(product_duration),
            'BounceRates': float(DEFAULT_VALUES['BounceRates']),
            'ExitRates': float(DEFAULT_VALUES['ExitRates']),
            'PageValues': float(page_values),
            'SpecialDay': float(DEFAULT_VALUES['SpecialDay']),
        }
        
        # 2. Crear el DataFrame a partir del diccionario
        final_input_df = pd.DataFrame([input_data_dict])

        # 3. Asegurar el ORDEN CORRECTO de las 13 columnas
        final_input_df = final_input_df[FULL_FEATURE_LIST]

        # 4. Realizar la Predicción
        try:
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(final_input_df)[:, 1][0]
                prediction = (prediction_proba >= 0.5) * 1 
            else:
                prediction = model.predict(final_input_df)[0]
                prediction_proba = None
            
            
            # 5. Mostrar el Resultado
            st.subheader("Resultado de la Predicción")
            
            if prediction == 1:
                st.balloons() # <--- Animación de CONFETI
                st.success("$$€€ ¡El modelo predice que **SÍ** hay una **Intención de Compra** (Revenue=True)! 💰") 
            else:
                st.info("📉 El modelo predice que **NO** hay una Intención de Compra (Revenue=False).")

            if prediction_proba is not None:
                st.metric(
                    label="Probabilidad de Compra (Revenue=True)",
                    value=f"{prediction_proba * 100:.2f} %"
                )
                st.progress(prediction_proba)

        except Exception as e:
            st.error(f"Ocurrió un error inesperado durante la predicción. Error: {e}")
            st.markdown("---")
            st.caption("Detalles para Depuración (DataFrame de Entrada Final):")
            st.dataframe(final_input_df)
            