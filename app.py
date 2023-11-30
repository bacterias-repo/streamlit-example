import streamlit as st

# Título de la aplicación
st.title("Cargador de Archivos .LAS")

# Widget para cargar un archivo .LAS
archivo_las = st.file_uploader("Selecciona un archivo .LAS", type=["las"])

if archivo_las is not None:
    # Obtenemos el nombre del archivo
    nombre_archivo = archivo_las.name
    
    # Imprimir mensaje de éxito
    st.success(f"Archivo {nombre_archivo} cargado correctamente")
