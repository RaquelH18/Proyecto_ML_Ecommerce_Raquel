![Ecommerce](Ecomerce.jpg)

## 🛒 Proyecto Final de Machine Learning: 
## Predicción de Intención de Compra en E-commerce
Autor: Raquel Hernández Lozano

Fecha: Noviembre 2025


Este proyecto de Data Science tiene como objetivo principal construir un **modelo de Machine Learning (Clasificación)** capaz de predecir si un visitante de una plataforma de comercio electrónico (e-commerce) realizará o no una compra, basándose en su comportamiento de navegación. Adicionalmente, se realiza una **segmentación de clientes (Clustering)** para identificar distintos grupos de comportamiento.

El modelo de predicción es crucial para que las empresas puedan optimizar sus estrategias de marketing, personalizar la experiencia del usuario y reducir la tasa de abandono del carrito.

### 🎯 Objetivos Específicos
1.  **Limpieza y Exploración (EDA):** Entender y preparar el dataset de comportamiento de usuarios.
2.  **Modelado Predictivo:** Entrenar, evaluar y seleccionar el mejor modelo de clasificación para predecir la intención de compra (`Revenue`).
3.  **Modelado Descriptivo:** Aplicar técnicas de *clustering* (e.g., K-Means) para segmentar a los visitantes.
4.  **Despliegue:** Crear una aplicación web interactiva con **Streamlit** para demostrar la capacidad predictiva del modelo.

---

## 📁 Estructura del Proyecto

El repositorio está organizado siguiendo una estructura común en proyectos de Machine Learning:

| Carpeta | Descripción | Contenido Relevante |
| :--- | :--- | :--- |
| `notebooks/` | Contiene los *Jupyter Notebooks* con el flujo completo de análisis. | `01_Fuentes.ipynb`, `02_LimpiezaEDA_OK.ipynb`, `03_Entrenamiento_Evaluacion_OK.ipynb` |
| `data/` | Almacena los datos en sus diferentes etapas. | `raw/online_shoppers_intention.csv` (Datos originales) |
| `models/` | Aquí se guardan los modelos entrenados y listos para ser usados. | `best_model_shoppers.pkl` (Modelo de Clasificación), `kmeans_model.pkl` (Modelo de Clustering) |
| `app_streamlit/` | Contiene el código Python para la aplicación web interactiva. | `app.py` |

---

## 📊 Dataset y Preprocesamiento

### Fuente de Datos
Se utilizó el conjunto de datos `online_shoppers_intention.csv`, que contiene datos recopilados de sesiones de 12.330 visitantes a un e-commerce, incluyendo variables como la duración de las visitas a diferentes tipos de páginas, información administrativa, tráfico y más.

La variable objetivo es **`Revenue`**, una variable booleana que indica si la sesión resultó en una compra (True) o no (False).

###  Proceso de Limpieza y EDA
El proceso de limpieza, transformación y Análisis Exploratorio de Datos (EDA) se detalla en el notebook **`02_LimpiezaEDA_OK.ipynb`**. Este incluye:
* Manejo de valores nulos y atípicos.
* Codificación de variables categóricas (Label Encoding).
* Análisis de la distribución de las variables y relación con la variable objetivo.
* Gestión del desbalance de clases

---

## 🧠 Modelado y Resultados

### 1. Modelo de Clasificación (Predicción de Compra)
El entrenamiento y la evaluación de los modelos de clasificación se encuentran en el notebook **`03_Entrenamiento_Evaluacion_OK.ipynb`**.

* **Algoritmos Probados:** Se evaluaron varios modelos de clasificación (ej. Regresión Logística, Random Forest, Gradient Boosting....).
* **Métrica Principal:** Dada la naturaleza del problema (predecir una compra exitosa), se priorizó la métrica **Recall** (sensibilidad) para minimizar los falsos negativos (no predecir una compra que sí ocurrirá).
* **Modelo Final:** El modelo con mejor rendimiento fue serializado como `best_model_shoppers.pkl`.

### 2. Modelo de Clustering (Segmentación)
Además de la predicción, se aplicó el algoritmo **K-Means** (`kmeans_model.pkl`) para identificar grupos de visitantes con patrones de navegación similares, lo que puede ser utilizado para estrategias de marketing diferenciadas.

---

##  Aplicación Web (Streamlit)

Se desarrolló una aplicación web interactiva en Python utilizando **Streamlit** para poner el modelo de clasificación a disposición del usuario.

El archivo **`app_streamlit/app.py`** permite a cualquier usuario introducir las características de una sesión de navegación y obtener una predicción instantánea sobre si esa sesión resultará en una compra o no.




