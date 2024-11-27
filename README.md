# Proyecto2
Clasificación Basada en Árboles
Objetivo: Realizar un análisis exploratorio de datos (EDA) inicial para al menos cuatro conjuntos de datos, diagnosticar y elegir una problemática específica para abordar (regresión, clasificación, clusterización, predicción). Entregar un repositorio con el dataset elegido, el EDA inicial y la problemática seleccionada.

Parte I: Búsqueda y Análisis de Conjuntos de Datos

Instrucciones

    Búsqueda de Conjuntos de Datos:
        Buscar al menos cuatro conjuntos de datos en plataformas como Kaggle, UCI Machine Learning Repository o cualquier otra fuente confiable.
        Asegurarse de que los conjuntos de datos seleccionados sean diversos y cubran diferentes dominios y tipos de datos.
    Análisis Exploratorio de Datos (EDA) Inicial:
        Realizar un EDA inicial para cada uno de los cuatro conjuntos de datos seleccionados.
        Incluir visualizaciones, análisis estadístico descriptivo, identificación de valores nulos y outliers.
        Documentar los hallazgos de cada EDA en un notebook de Jupyter.
    Diagnóstico y Selección de Problema:
        Basándose en el EDA inicial, diagnosticar las principales características y desafíos de cada conjunto de datos.
        Elegir una problemática específica para abordar (regresión, clasificación, clusterización, predicción).
        Justificar la elección del problema y explicar por qué es relevante y desafiante.
    Creación del Repositorio en GitHub:
        Crear un repositorio en GitHub para el Proyecto 2.
        Incluir el EDA inicial de los cuatro conjuntos de datos en notebooks separados.
        Incluir una carpeta para el dataset elegido con su EDA correspondiente.
        Documentar la problemática seleccionada y justificar la elección en un archivo README.md.

Detalles del EDA Inicial

    Descripción del Conjunto de Datos:
        Breve descripción de cada conjunto de datos, incluyendo la fuente, el tamaño y las variables.
    Análisis Estadístico Descriptivo:
        Calcular estadísticas descriptivas básicas (media, mediana, desviación estándar, etc.) para las variables numéricas.
        Analizar la distribución de las variables categóricas.
    Visualizaciones:
        Crear visualizaciones para entender la distribución de las variables (histogramas, gráficos de barras, box plots, etc.).
        Visualizar las correlaciones entre variables (mapa de calor de correlación).
    Identificación de Valores Nulos y Outliers:
        Detectar valores nulos y discutir cómo podrían ser tratados.
        Identificar outliers y evaluar su impacto potencial en el análisis.
    Resumen de Hallazgos:
        Resumir los principales hallazgos de cada EDA, destacando las características y desafíos únicos de cada conjunto de datos.

        ---

        Parte II: Preprocesamiento y Optimización

Objetivo: Realizar el preprocesamiento de datos y la optimización de modelos de machine learning para el conjunto de datos seleccionado. La meta es elegir la técnica de machine learning más adecuada y optimizar sus hiperparámetros para obtener el mejor rendimiento posible.


Instrucciones Detalladas

Parte 1: Preprocesamiento de Datos

    Limpieza de Datos:
        Tratar los valores nulos utilizando técnicas adecuadas (imputación, eliminación, etc.).
        Manejar los outliers mediante técnicas de filtrado o transformación.
    Transformación de Columnas:
        Utilizar ColumnTransformer para aplicar transformaciones específicas a diferentes columnas.
        Realizar codificación de variables categóricas utilizando técnicas como One-Hot Encoding.
        Escalar las variables numéricas usando StandardScaler u otros métodos de normalización.
    Creación de Pipelines:
        Crear pipelines utilizando Pipeline de sklearn para automatizar el preprocesamiento de datos y asegurar la reproducibilidad.
        Incluir todos los pasos de preprocesamiento en el pipeline.


Parte 2: Selección de Técnica de Machine Learning

    Entrenamiento Inicial:
        Entrenar múltiples modelos de machine learning (por ejemplo, Regresión Lineal, KNN, Árbol de Decisión, Random Forest, XGBoost, LGBM).
        Evaluar los modelos utilizando validación cruzada y seleccionar el modelo con el mejor rendimiento inicial.
    Comparación de Modelos:
        Comparar los modelos utilizando métricas de rendimiento relevantes (exactitud, precisión, recall, F1-Score, ROC-AUC, etc.).
        Seleccionar la técnica de machine learning más adecuada basándose en las métricas y la naturaleza del problema.


Parte 3: Optimización de Hiperparámetros

    GridSearchCV:
        Implementar GridSearchCV para realizar una búsqueda exhaustiva de los mejores hiperparámetros para el modelo seleccionado.
        Definir el espacio de búsqueda para los hiperparámetros relevantes.
    RandomizedSearchCV:
        Implementar RandomizedSearchCV para realizar una búsqueda aleatoria de los mejores hiperparámetros, especialmente útil si el espacio de búsqueda es grande.
    Optuna:
        Implementar Optuna para una optimización avanzada de los hiperparámetros, aprovechando técnicas como la optimización bayesiana y el pruning.
    Evaluación de Modelos Optimizados:
        Entrenar el modelo con los mejores hiperparámetros encontrados y evaluar su rendimiento en el conjunto de prueba.
        Comparar el rendimiento del modelo optimizado con el modelo inicial.


Parte 4: Documentación y Entrega

    Documentación del Proceso:
        Documentar todos los pasos del preprocesamiento, selección de técnica y optimización en un notebook de Jupyter.
        Incluir explicaciones detalladas y justificaciones para cada decisión tomada.
    Subida a GitHub:
        Actualizar el repositorio de GitHub con los notebooks de preprocesamiento, selección de técnica y optimización.
        Incluir los resultados de la optimización y la comparación de modelos.
        Crear un tag de liberación (v2.0.0) para esta versión del proyecto.
