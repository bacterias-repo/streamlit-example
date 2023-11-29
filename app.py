import streamlit as st
import lasio
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el archivo LAS desde Streamlit o desde GitHub
st.title("Visualización de Regresión")

uploaded_file = st.file_uploader("Cargar archivo LAS", type=["las"])

if uploaded_file is not None:
    # Si se cargó un archivo LAS, usar ese archivo
    las = lasio.read(uploaded_file)
    df = las.df()
else:
    # Si no se cargó un archivo, cargar el archivo LAS desde GitHub
    las_file_url = "https://raw.githubusercontent.com/bacterias-repo/streamlit-example/master/Par%C3%A1metros%20Akac%C3%ADas%2078%2015-08-2022%200516h.las"
    response = requests.get(las_file_url)
    las_content = io.StringIO(response.text)
    las = lasio.read(las_content, autodetect_encoding=True, ignore_header_errors=True, encoding='latin1', engine='normal')
    df = las.df()

# Visualización de los datos
st.header("Visualización de datos")
st.write("Comportamiento de la variable de interés (ROP_T)")

xpoints = [i for i in range(df.shape[0])]
ypoints = df.ROP_T.values

fig = plt.figure(figsize=(15, 5))
plt.scatter(xpoints, ypoints)
plt.title('Tiempo vs ROP')
st.pyplot(fig)

# Dividir los datos en características y variable objetivo
variables = ['DBTM', 'DMEA', 'BLKPOS', 'ECD_T', 'HKLA', 'MDIA', 'FLOWIN_T', 'RPM_T', 'SPP_T', 'TVA', 'TOR_T', 'WOB_T']
datosX = df[variables]
datosY = df.ROP_T

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datosX.values, datosY.values, test_size=0.3, random_state=0)

# Entrenar y evaluar un modelo LSTM
st.header("Entrenamiento y evaluación del modelo LSTM")

# Crear el modelo LSTM
model = keras.Sequential()
model.add(keras.layers.LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32)

# Realizar predicciones
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predictions = model.predict(X_test_reshaped)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar la arquitectura y parámetros de entrenamiento
st.subheader("Arquitectura y parámetros de entrenamiento del modelo LSTM")
st.text(model.summary())

st.subheader("Métricas de rendimiento del modelo LSTM")
st.write(f"MSE: {mse}")
st.write(f"R2 Score: {r2}")

# Gráficas de aprendizaje
st.subheader("Gráficas de aprendizaje del modelo LSTM")
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label='Valores reales', marker='o')
plt.plot(predictions[:100], label='Predicciones', marker='x')
plt.xlabel('Muestras')
plt.ylabel('ROP_T')
plt.legend()
st.pyplot()

# Resultados del modelo LSTM
st.header("Resultados del modelo LSTM")

# Realizar predicciones en todo el conjunto de prueba
predictions = model.predict(X_test_reshaped)

# Tomar las primeras 100 muestras para la visualización
xpoints = [i for i in range(100)]
ypoints_pred = predictions[:100]
ypoints_real = y_test[:100]

fig = plt.figure(figsize=(8, 5))
plt.scatter(xpoints, ypoints_real, s=50, edgecolors='black', c='yellow', label='Valores reales')
plt.scatter(xpoints, ypoints_pred, s=20, c='red', label='Predicciones')

MSE = mean_squared_error(y_test, predictions)
R2 = r2_score(y_test, predictions)
RMSE = np.sqrt(MSE)

plt.title(f'Predicciones LSTM -- R2: {R2:.4f} -- MSE: {MSE:.4f} -- RMSE: {RMSE:.4f}')
plt.xlabel('Muestras')
plt.ylabel('ROP_T')
plt.legend()
st.pyplot(fig)

# Puedes ejecutar la aplicación web con Streamlit ejecutando el siguiente comando en tu terminal:
# streamlit run app.py
