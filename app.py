import streamlit as st
import lasio
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from io import StringIO

# Título de la aplicación
st.title("Deep Learning para la predicción de la ROP")

# Widget para cargar un archivo .LAS
archivo_las = st.file_uploader("Selecciona un archivo .LAS", type=["las"])

if archivo_las is not None:
    # Leer el contenido del archivo LAS con lasio
    bytes_data = archivo_las.read()
    str_io = StringIO(bytes_data.decode('Windows-1252'))
    las_file = lasio.read(str_io)
    datos = las_file.df()
    datos = datos.dropna()

    # se visualiza el comportamiento de las variable de interés
    # se crea un vector de igual número de elementos de todos las muestras
    xpoints = [i for i in range(datos.shape[0])]
    # se toman las medidas de viscosidad
    ypoints = datos.ROP_T.values
    fig = plt.figure(figsize=(15, 5))
    # se pintan
    plt.scatter(xpoints, ypoints)
    plt.title('Tiempo vs ROP')
    st.pyplot(fig)

    variables = ['DBTM', 'DMEA', 'BLKPOS', 'ECD_T', 'HKLA', 'MDIA', 'FLOWIN_T', 'RPM_T', 'SPP_T', 'TVA', 'TOR_T', 'WOB_T']

    # se toman los datos de las variables que se escogieron como variables predictoras
    datosX = datos[variables]
    # se extrae la variable a predecir
    datosY = datos.ROP_T

    # se dividen los datos en 70%, 30% de manera aleatoria
    X_train, X_test, y_train, y_test = train_test_split(datosX.values, datosY.values, test_size=0.3, random_state=0)

    # se instancian los métodos de regresión
    regresores = [
        ("Extra_Trees_Regressor", ExtraTreesRegressor()),
        ("Gradient_Boosting_Regressor", GradientBoostingRegressor()),
        ("Arboles_de_decision", DecisionTreeRegressor()),
        ("Bosques_aleatorios", RandomForestRegressor())
    ]

    # Dropdown para seleccionar el modelo
    modelo_seleccionado = st.selectbox("Selecciona un modelo", [nombre for nombre, _ in regresores])

    for nombre, regresor in regresores:
        if nombre == modelo_seleccionado:
            # se entrena el modelo seleccionado con los datos de entrenamiento
            regresor.fit(X_train, y_train)
            # se realizan las predicciones para el modelo seleccionado en los datos de prueba
            predicciones = regresor.predict(X_test)

            # se crea el tamaño de la figura
            fig = plt.figure(figsize=(8, 5))

            # se calculan los parámetros del modelo
            MSE = mean_squared_error(y_test, predicciones)
            R2 = r2_score(y_test, predicciones)
            RMSE = np.sqrt(MSE)

            # se pinta el cuadro de convecciones
            plt.scatter(xpoints, y_test, s=50, edgecolors='black', c='yellow', label='reales')
            plt.scatter(xpoints, predicciones, s=20, c='red', label='predicciones')

            # se muestra información sobre el modelo seleccionado
            plt.title(f'Pred {nombre} -- R2:{R2} -- MSE:{MSE} -- RMSE:{RMSE}')
            plt.legend()

            # guardar el modelo en disco
            filename = nombre + 'modelo.sav'
            pickle.dump(regresor, open(filename, 'wb'))
            print(filename)
            st.pyplot(fig)
