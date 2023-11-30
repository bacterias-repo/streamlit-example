import streamlit as st
import lasio
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Cargar modelos previamente entrenados
def cargar_modelos():
    modelos = {}
    regresores = ["Extra_Trees_Regressor", "Gradient_Boosting_Regressor", "Arboles_de_decision", "Bosques_aleatorios"]
    
    for nombre in regresores:
        filename = nombre + 'modelo.sav'
        modelos[nombre] = pickle.load(open(filename, 'rb'))
    
    return modelos

# Interfaz de usuario con Streamlit
def main():
    st.title("Predicciones de ROP (Rate of Penetration)")

    # Cargar modelos
    modelos = cargar_modelos()

    uploaded_file = st.file_uploader("Cargar archivo LAS", type=["las"])

    if uploaded_file is not None:
        # Leer el archivo LAS
        las = lasio.read(uploaded_file, autodetect_encoding=True, ignore_header_errors=True, encoding='latin1', engine='normal')
        datos = las.df()

        # Filtrar y limpiar datos
        datos = datos.dropna()
        variables = ['DBTM', 'DMEA', 'BLKPOS', 'ECD_T', 'HKLA', 'MDIA', 'FLOWIN_T', 'RPM_T', 'SPP_T', 'TVA', 'TOR_T', 'WOB_T']
        datosX = datos[variables]
        datosY = datos['ROP_T']

        # Predicciones
        st.header("Selecciona un modelo para hacer predicciones:")
        modelo_seleccionado = st.selectbox("Modelo:", ["Extra_Trees_Regressor", "Gradient_Boosting_Regressor", "Arboles_de_decision", "Bosques_aleatorios"])

        if st.button("Hacer predicciones"):
            modelo = modelos[modelo_seleccionado]
            predicciones = modelo.predict(datosX)
            st.write("Predicciones de ROP:")
            st.write(predicciones)

            # Métricas de evaluación
            MSE = mean_squared_error(datosY, predicciones)
            R2 = r2_score(datosY, predicciones)
            RMSE = np.sqrt(MSE)
            st.write(f"MSE: {MSE}")
            st.write(f"R2: {R2}")
            st.write(f"RMSE: {RMSE}")

        # Visualización de datos
        st.header("Visualización de datos:")
        st.write("Datos de entrada:")
        st.write(datosX.head())
        st.write("Datos de salida (ROP_T):")
        st.write(datosY.head())

        # Puedes agregar gráficos adicionales aquí según tus necesidades

main()
