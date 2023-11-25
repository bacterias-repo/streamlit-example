import streamlit as st
import lasio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
import numpy as np

# Cargar el archivo LAS y convertirlo a un DataFrame
# URL del archivo LAS en GitHub
las_file_url = "https://raw.githubusercontent.com/bacterias-repo/streamlit-example/master/Par%C3%A1metros%20Akac%C3%ADas%2078%2015-08-2022%200516h.las"

# Descargar el archivo LAS desde GitHub
response = requests.get(las_file_url)

# Verificar si la descarga fue exitosa
if response.status_code == 200:
    # Crear un objeto StringIO a partir del contenido descargado
    las_content = io.StringIO(response.text)

    # Cargar el archivo LAS y convertirlo a un DataFrame
    las = lasio.read(las_content, autodetect_encoding=True, ignore_header_errors=True, encoding='latin1', engine='normal')
    df = las.df()
    datos = df.copy()
    datos = datos.dropna()

# Crear una página web con Streamlit
st.title("Visualización de Regresión")

# Visualización de los datos
st.header("Visualización de datos")
st.write("Comportamiento de la variable de interés (ROP_T)")

xpoints = [i for i in range(datos.shape[0])]
ypoints = datos.ROP_T.values

fig = plt.figure(figsize=(15, 5))
plt.scatter(xpoints, ypoints)
plt.title('Tiempo vs ROP')
st.pyplot(fig)

# Dividir los datos en características y variable objetivo
variables = ['DBTM', 'DMEA', 'BLKPOS', 'ECD_T', 'HKLA', 'MDIA', 'FLOWIN_T', 'RPM_T', 'SPP_T', 'TVA', 'TOR_T', 'WOB_T']
datosX = datos[variables]
datosY = datos.ROP_T

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datosX.values, datosY.values, test_size=0.3, random_state=0)

# Entrenar y evaluar modelos de regresión
st.header("Entrenamiento y evaluación de modelos")

regresores = [
    ("Extra_Trees_Regressor", ExtraTreesRegressor()),
    ("Gradient_Boosting_Regressor", GradientBoostingRegressor()),
    ("Arboles_de_decision", DecisionTreeRegressor()),
    ("Bosques_aleatorios", RandomForestRegressor())
]

for nombre, regresor in regresores:
    regresor.fit(X_train, y_train)
    predicciones = regresor.predict(X_test)
    xpoints = [i for i in range(100)]
    ypoints = predicciones[:100]
    reales = y_test[:100]

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(xpoints, reales, s=50, edgecolors='black', c='yellow', label='reales')
    plt.scatter(xpoints, ypoints, s=20, c='red', label='predicciones')

    MSE = mean_squared_error(y_test, predicciones)
    R2 = r2_score(y_test, predicciones)
    RMSE = np.sqrt(MSE)

    plt.title('Predicciones '+nombre+" -- R2:"+str(R2)+" -- MSE:"+str(MSE)+" -- RMSE:"+str(RMSE))
    plt.legend()
    st.pyplot(fig)

    # Guardar el modelo en disco
    filename = nombre + 'modelo.sav'
    pickle.dump(regresor, open(filename, 'wb'))
    st.write(f"Modelo guardado como '{filename}'")

# Puedes ejecutar la aplicación web con Streamlit ejecutando el siguiente comando en tu terminal:
# streamlit run app.py
