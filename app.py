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
import os

# Título de la aplicación
st.title("Deep Learning para la predicción de la ROP")

# Widget para cargar un archivo .LAS
archivo_las = st.file_uploader("Selecciona un archivo .LAS", type=["las"])

if archivo_las is not None:
    # Obtenemos el nombre del archivo
    nombre_archivo = archivo_las.name

    # Guardar el archivo .LAS en disco
    with open(nombre_archivo, "rb") as f:
        f.write(archivo_las.read())

    # Imprimir mensaje de éxito
    st.success(f"Archivo {nombre_archivo} cargado correctamente")

    # Cargar los datos del archivo .LAS utilizando lasio
    with open(nombre_archivo, "rb") as f:
        las_file = lasio.read(f)
    
    # Convertir los datos a un DataFrame de pandas
    datos = las_file.df()

    # Resto del código para visualizar y entrenar modelos...
    os.remove(nombre_archivo)  # Eliminar el archivo después de su uso

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
    # no se ponen tildes en el nombre para poder guardar el modelo con esta misma variable
    regresores = [("Extra_Trees_Regressor", ExtraTreesRegressor()),
                  ("Gradient_Boosting_Regressor", GradientBoostingRegressor()),
                  ("Arboles_de_decision", DecisionTreeRegressor()),
                  ("Bosques_aleatorios", RandomForestRegressor())]
    # se toman 100 puntos aleatorios de los datos de test para poder visualizar el rendimiento de los métodos
    points = [random.randint(1, X_test.shape[0]) for i in range(100)]
    for nombre, regresor in regresores:
        # se entrena cada método con los datos de test
        regresor.fit(X_train, y_train)
        # se realizan las predicciones para cada método en los datos de test
        predicciones = regresor.predict(X_test)
        # se crean los puntos del eje x para plotear, 100 porque fue el número de puntos que se escogió para ver
        xpoints = [i for i in range(100)]
        # se escogen las predicciones correspondientes a los 100 puntos
        ypoints = predicciones[points]
        # se toman los mismos 100 puntos para comparar con las predicciones
        reales = y_test[points]
        # se crea el tamaño de la figura
        fig = plt.figure(figsize=(8, 5))
        # se pintan los puntos reales
        plt.scatter(xpoints, reales, s=50, edgecolors='black', c='yellow', label='reales')
        # se pintan las predicciones
        plt.scatter(xpoints, ypoints, s=20, c='red', label='predicciones')

        # se calcula el score r cuadrado entre las predicciones y los datos reales
        # rscore = r2_score(y_test, predicciones);
        # plt.title('Predicciones '+nombre+" -- R2:"+str(rscore));

        MSE = mean_squared_error(y_test, predicciones)  # Error cuadrático medio para la red neuronal
        R2 = r2_score(y_test, predicciones)  # Coeficiente de determinación para la red neuronal
        RMSE = np.sqrt(MSE)  # Raíz del error cuadrático medio para la red neuronal
        plt.title('Pred ' + nombre + " -- R2:" + str(R2) + " -- MSE:" + str(MSE) + " -- RMSE:" + str(RMSE))

        # pinta el cuadro de convecciones
        plt.legend()
        # guardar el modelo en disco
        filename = nombre + 'modelo.sav'
        pickle.dump(regresor, open(filename, 'wb'))
        print(filename)
        st.pyplot(fig)
