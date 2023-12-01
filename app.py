from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import StringIO
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pickle
import lasio
import os

# -----------------------------------------------------------------------------------------------------------------
# FUNCIONES AUXILIARES --------------------------------------------------------------------------------------------
def eda_datos(datos):
    """
    Realiza un análisis exploratorio de datos (EDA) en el DataFrame 'datos'.

    :param datos: DataFrame de Pandas con los datos a analizar.
    """
    if datos is not None:
        st.write("## Análisis Exploratorio de Datos")

        # Datos completos
        st.write("### Visualización de la ROP")
        xpoints = [i for i in range(datos.shape[0])]
        ypoints = datos.ROP_T.values
        fig = plt.figure(figsize=(15, 5))
        # se pintan
        plt.scatter(xpoints, ypoints)
        plt.title('Tiempo vs ROP')
        st.pyplot(fig)

        # Estadísticas descriptivas
        st.write("### Estadísticas Descriptivas")
        st.dataframe(datos.describe())

        # Tipos de variables
        st.write("### Tipos de Variables")
        st.dataframe(datos.dtypes)

        # Dropdown para estadísticas descriptivas de cada columna
        columna_seleccionada = st.selectbox("Selecciona una columna para más detalles", datos.columns)
        st.write(datos[columna_seleccionada].describe())

        # Histograma, Boxplot y Q-Q plot
        st.write("### Visualizaciones de la columna seleccionada")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(datos[columna_seleccionada], kde=True, ax=ax[0])
        sns.boxplot(y=datos[columna_seleccionada], ax=ax[1])
        stats.probplot(datos[columna_seleccionada].dropna(), dist="norm", plot=ax[2])
        st.pyplot(fig)

        # Correlaciones
        st.write("### Correlaciones")
        corr = datos.corr()
        
        # Crear una figura y un eje con Matplotlib
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        
        # Mostrar la figura en Streamlit
        st.pyplot(fig)


        # Datos faltantes
        st.write("### Datos Faltantes")
        st.dataframe(datos.isnull().sum())

        # Filas duplicadas
        st.write("### Filas Duplicadas")
        st.write("Número de filas duplicadas:", datos.duplicated().sum())

def imputacion_knn(datos):
    imputer = KNNImputer(n_neighbors=2)
    datos_imputados = imputer.fit_transform(datos.select_dtypes(include=[np.number]))
    return pd.DataFrame(datos_imputados, columns=datos.columns)

def eliminar_valores_faltantes(datos):
    return datos.dropna()

def eliminar_columnas_poco_representativas(datos):
    # Ejemplo: eliminar columnas con una sola categoría
    for col in datos.columns:
        if datos[col].nunique() <= 1:
            datos.drop(col, axis=1, inplace=True)
    return datos.dropna()

def eliminar_por_correlacion(datos):
    corr_matrix = datos.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return datos.drop(to_drop, axis=1).dropna()

def imputar_knn_eliminar_outliers(datos):
    datos = imputacion_knn(datos)
    z_scores = np.abs(stats.zscore(datos.select_dtypes(include=[np.number])))
    return datos[(z_scores < 3).all(axis=1)]

def eliminar_nan_outliers(datos):
    datos = eliminar_valores_faltantes(datos)
    z_scores = np.abs(stats.zscore(datos.select_dtypes(include=[np.number])))
    return datos[(z_scores < 3).all(axis=1)]

def eliminar_columnas_nan_outliers(datos):
    datos = eliminar_columnas_poco_representativas(datos)
    z_scores = np.abs(stats.zscore(datos.select_dtypes(include=[np.number])))
    return datos[(z_scores < 3).all(axis=1)]

def eliminar_correlacion_nan_outliers(datos):
    datos = eliminar_por_correlacion(datos)
    z_scores = np.abs(stats.zscore(datos.select_dtypes(include=[np.number])))
    return datos[(z_scores < 3).all(axis=1)]
    
# FIN DE FUNCIONES AUXILIARES -------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------
# ARQITECTURA DE RED NEURONAL -------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Crear un modelo secuencial
model = Sequential()

# Agregar capas ocultas personalizadas
model.add(Dense(128, activation='relu', input_shape=(12,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Capa de salida con activación lineal para regresión

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# ARQITECTURA DE RED NEURONAL -------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


# Título de la aplicación
st.title("Deep Learning para la predicción de la ROP")

# LIMPIAR LOS DATOS
metodo = st.selectbox(
    "Elige un método de limpieza:",
    ["Imputación KNN", "Eliminar NaN", "Eliminar Columnas y NaN",
     "Eliminar por Correlación y NaN", "Imputar KNN y Eliminar Outliers",
     "Eliminar NaN y Outliers", "Eliminar Columnas, NaN y Outliers",
     "Eliminar Correlación, NaN y Outliers"],
    index=0  # Esto establece "Imputación KNN" como la opción por defecto
)


# Widget para cargar un archivo
archivo = st.file_uploader("Selecciona un archivo (.LAS, .XLSX, .CSV)", type=["las", "xlsx", "csv"])

if archivo is not None:
    # Verificar tipo de archivo y procesar
    if archivo.name.endswith('.las'):
        # Procesar archivo LAS
        bytes_data = archivo.read()
        str_io = StringIO(bytes_data.decode('Windows-1252'))
        las_file = lasio.read(str_io)
        datos = las_file.df()
    elif archivo.name.endswith(('.xlsx', '.csv')):
        # Procesar archivo Excel o CSV
        datos = pd.read_excel(archivo) if archivo.name.endswith('.xlsx') else pd.read_csv(archivo)


    # MOSTRAR LA EXPLORACIÓN DE LOS DATOS
    datos = datos.fillna(value=np.nan)
    eda_datos(datos)

    if metodo == "Método 1":
        datos = imputacion_knn(datos)
    elif metodo == "Método 2":
        datos = eliminar_valores_faltantes(datos)
    elif metodo == "Método 3":
        datos = eliminar_columnas_poco_representativas(datos)
    elif metodo == "Método 4":
        datos = eliminar_por_correlacion(datos)
    elif metodo == "Método 5":
        datos = imputar_knn_eliminar_outliers(datos)
    elif metodo == "Método 6":
        datos = eliminar_nan_outliers(datos)
    elif metodo == "Método 7":
        datos = eliminar_columnas_nan_outliers(datos)
    elif metodo == "Método 8":
        datos = eliminar_correlacion_nan_outliers(datos)
    st.write("Datos después de la limpieza:")
    st.dataframe(datos)
    datos = datos.dropna()

    st.write("Entrenamiento de modelos")
    # ENTRENAMIENTO DE LOS DIFRENTES MODELOS
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
                  ("Bosques_aleatorios", RandomForestRegressor()),
                  ("Red_neuronal", model)]
    # se toman 100 puntos aleatorios de los datos de test para poder visualizar el rendimiento de los métodos
    points = [random.randint(1, X_test.shape[0]) for i in range(100)]

    # Dropdown para seleccionar el modelo
    modelo_seleccionado = st.selectbox("Selecciona un modelo", [nombre for nombre, _ in regresores])

    for nombre, regresor in regresores:
        if nombre == modelo_seleccionado:
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
            st.pyplot(fig)
