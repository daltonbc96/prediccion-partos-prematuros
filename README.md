# Análisis y Modelado de Datos del Sistema de Información Perinatal (SIP) para la Predicción de Partos Prematuros

## Descripción del Proyecto

Este proyecto se centra en el desarrollo de modelos de machine learning utilizando datos del Sistema de Información Perinatal (SIP) para mejorar la calidad de la atención prestada a mujeres y niños en la región de las Américas. El objetivo es utilizar estos datos para predecir la semana de parto, lo cual es crucial para planificar intervenciones médicas adecuadas y mejorar los resultados de salud materno-infantil.

El proyecto es una iniciativa conjunta de la OPS, a través de su departamento de *Evidence and Intelligence for Action in Health* (EIH), y el *Latin American Center of Perinatology, Women and Reproductive Health* (CLAP/WR). Se enfoca en la estratificación del riesgo de la población en cuanto a la ocurrencia de nacimientos prematuros.

## Estructura del Proyecto

El proyecto se desarrolla en cuatro etapas principales:

1. **Carga de Información del SIP y Procesamiento de Datos**: Se inicia con la carga de las bibliotecas necesarias y los datos originales del SIP, junto con un diccionario de datos organizado en un archivo JSON. Este paso establece una base sólida para el análisis de datos.

2. **Análisis y Tratamiento de Datos Faltantes**: Se enfoca en el análisis de datos faltantes y su tratamiento. Se utilizan técnicas avanzadas como la imputación múltiple y la transformación de variables categóricas para preparar los datos para el modelado.

3. **Controles de Asociación**: En esta etapa, se llevan a cabo análisis para identificar y comprender las asociaciones entre las diferentes variables y la semana de parto.

4. **Evaluación de Modelos de Machine Learning para Problemas de Regresión**: Se prueban y comparan diferentes algoritmos de machine learning para predecir la semana de parto. Se utilizan técnicas de validación cruzada y optimización bayesiana para encontrar los mejores modelos y hiperparámetros.

## Metodología

El proyecto aplica una metodología rigurosa para el desarrollo de modelos de machine learning:

- **Preprocesamiento de Datos**: Incluye la carga, limpieza, y transformación de los datos del SIP.
- **Análisis Exploratorio de Datos (EDA)**: Se realiza un análisis detallado de los datos para entender sus características y prepararlos para el modelado.
- **Optimización de Modelos**: Se utilizan técnicas de optimización bayesiana, como BayesSearchCV, para ajustar los hiperparámetros de los modelos.
- **Validación Cruzada**: Se implementa un enfoque de validación cruzada basado en el nivel de participante para evaluar el rendimiento de los modelos.
- **Evaluación de Modelos**: Se comparan varios modelos utilizando métricas como R2, MAE y RMSE para seleccionar el mejor modelo.

## Resultados y Conclusiones

El proyecto tiene como resultado un conjunto de modelos de machine learning capaces de predecir la semana de parto con precisión. Estos modelos pueden ser de gran ayuda para los profesionales sanitarios en la región, permitiendo intervenciones más efectivas y mejorando la salud materno-infantil.

## Cómo Utilizar

Este repositorio contiene notebooks de Jupyter que detallan cada paso del proyecto. Para replicar o extender este trabajo, se recomienda seguir la secuencia de los notebooks y adaptar los códigos y análisis según sea necesario para otros conjuntos de datos o preguntas de investigación.
