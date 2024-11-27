# **Proyecto 2: Clasificación Basada en Árboles**

## **Objetivo**
Realizar un análisis exploratorio de datos (EDA) inicial para al menos cuatro conjuntos de datos, diagnosticar y elegir una problemática específica para abordar (regresión, clasificación, clusterización, predicción). Entregar un repositorio con el dataset elegido, el EDA inicial y la problemática seleccionada.

---

## **Parte I: Búsqueda y Análisis de Conjuntos de Datos**

### **Instrucciones**

#### **Búsqueda de Conjuntos de Datos**
- Buscar al menos **cuatro conjuntos de datos** en plataformas como:
  - Kaggle
  - UCI Machine Learning Repository
  - Cualquier otra fuente confiable
- Asegurarse de que los conjuntos de datos seleccionados sean:
  - Diversos
  - Cubran diferentes dominios y tipos de datos

#### **Análisis Exploratorio de Datos (EDA) Inicial**
- Realizar un EDA inicial para cada uno de los cuatro conjuntos de datos seleccionados:
  - Visualizaciones
  - Análisis estadístico descriptivo
  - Identificación de valores nulos y outliers
- Documentar los hallazgos de cada EDA en un **notebook de Jupyter**.

#### **Diagnóstico y Selección de Problema**
- Diagnosticar las principales características y desafíos de cada conjunto de datos.
- Elegir una problemática específica para abordar:
  - Regresión
  - Clasificación
  - Clusterización
  - Predicción
- Justificar la elección y explicar por qué es relevante y desafiante.

#### **Creación del Repositorio en GitHub**
- Crear un repositorio para el **Proyecto 2**:
  - Incluir el EDA inicial de los cuatro conjuntos de datos en notebooks separados.
  - Incluir una carpeta para el dataset elegido con su EDA correspondiente.
  - Documentar la problemática seleccionada en un archivo `README.md`.

---

### **Detalles del EDA Inicial**

#### **Descripción del Conjunto de Datos**
- Breve descripción: 
  - Fuente
  - Tamaño
  - Variables

#### **Análisis Estadístico Descriptivo**
- Calcular estadísticas descriptivas básicas:
  - Media, mediana, desviación estándar
- Analizar la distribución de variables categóricas.

#### **Visualizaciones**
- Visualizar la distribución de variables:
  - Histogramas
  - Gráficos de barras
  - Box plots
- Visualizar correlaciones entre variables:
  - Mapa de calor de correlación

#### **Identificación de Valores Nulos y Outliers**
- Detectar valores nulos y proponer estrategias de tratamiento.
- Identificar outliers y evaluar su impacto.

#### **Resumen de Hallazgos**
- Resumir los principales hallazgos:
  - Características y desafíos únicos de cada conjunto de datos.

---

## **Parte II: Preprocesamiento y Optimización**

### **Objetivo**
Realizar el preprocesamiento de datos y la optimización de modelos de machine learning para el conjunto de datos seleccionado. Elegir la técnica más adecuada y optimizar los hiperparámetros para obtener el mejor rendimiento.

---

### **Instrucciones Detalladas**

#### **Parte 1: Preprocesamiento de Datos**

##### **Limpieza de Datos**
- Tratar valores nulos:
  - Imputación
  - Eliminación
- Manejar outliers:
  - Filtrado
  - Transformación

##### **Transformación de Columnas**
- Usar `ColumnTransformer` para transformaciones específicas.
- Codificar variables categóricas con **One-Hot Encoding**.
- Escalar variables numéricas con **StandardScaler** o métodos de normalización.

##### **Creación de Pipelines**
- Usar `Pipeline` para:
  - Automatizar el preprocesamiento
  - Asegurar reproducibilidad

---

#### **Parte 2: Selección de Técnica de Machine Learning**

##### **Entrenamiento Inicial**
- Entrenar múltiples modelos:
  - Regresión Lineal
  - KNN
  - Árbol de Decisión
  - Random Forest
  - XGBoost
  - LightGBM
- Evaluar con validación cruzada y seleccionar el mejor rendimiento inicial.

##### **Comparación de Modelos**
- Usar métricas relevantes:
  - Exactitud, precisión, recall, F1-Score, ROC-AUC
- Seleccionar la técnica más adecuada según las métricas.

---

#### **Parte 3: Optimización de Hiperparámetros**

##### **GridSearchCV**
- Realizar búsqueda exhaustiva de hiperparámetros.

##### **RandomizedSearchCV**
- Realizar búsqueda aleatoria para espacios de búsqueda grandes.

##### **Optuna**
- Implementar optimización avanzada con:
  - Optimización bayesiana
  - Pruning

##### **Evaluación de Modelos Optimizados**
- Entrenar con los mejores hiperparámetros.
- Evaluar rendimiento en el conjunto de prueba.
- Comparar rendimiento optimizado vs. inicial.

---

#### **Parte 4: Documentación y Entrega**

##### **Documentación del Proceso**
- Documentar todos los pasos en un notebook de Jupyter:
  - Preprocesamiento
  - Selección de técnica
  - Optimización
- Justificar cada decisión.

##### **Subida a GitHub**
- Actualizar el repositorio con:
  - Notebooks de preprocesamiento, selección y optimización.
  - Resultados de optimización y comparación de modelos.
- Crear un **tag de liberación**: `v2.0.0`.

---
