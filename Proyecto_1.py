import streamlit as st
import pandas as pd
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Expresiones regulares para identificar formatos de fecha
formatos_fecha = [
    r'\d{4}-\d{2}-\d{2}',           # Formato 'YYYY-MM-DD'
    r'\d{2}-\d{2}-\d{4}',           # Formato 'DD-MM-YYYY'
    r'\d{4}/\d{2}/\d{2}',           # Formato 'YYYY/MM/DD'
    r'\d{2}/\d{2}/\d{4}',           # Formato 'DD/MM/YYYY'
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Formato 'YYYY-MM-DD HH:mm:ss'
    r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}',   # Formato 'YYYY/MM/DD HH:mm:ss'
    r'\d{1,2}/\d{1,2}/\d{4}'         # Formato 'M/D/YYYY'
]

# Función para identificar formatos de fecha
def identificar_formato_fecha(valor):
    for formato in formatos_fecha:
        if re.match(formato, valor):
            return True
    return False

# Función para mostrar el gráfico de Matplotlib
def mostrar_matplotlib_fig(fig):
    st.pyplot(fig)

# Página de INICIO
def page_inicio():
    st.title("Proyecto Product Development - Fase I")
    st.markdown("---")
    st.write("¡Bienvenido a la sección de Inicio! Aquí puedes cargar un archivo .csv o .xlsx para analizar.")

    uploaded_file = st.file_uploader("Cargar archivo", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]

        if file_extension.lower() == ".csv":
            st.write("Archivo CSV cargado exitosamente.")
            st.session_state.df = pd.read_csv(uploaded_file)

        elif file_extension.lower() == ".xlsx":
            st.write("Archivo Excel cargado exitosamente.")
            st.session_state.df = pd.read_excel(uploaded_file)

        else:
            st.error("Formato de archivo no compatible. Por favor, carga un archivo CSV o Excel.")

        # Identificar y convertir columnas a tipo fecha
        for columna in st.session_state.df.columns:
            if st.session_state.df[columna].apply(lambda x: identificar_formato_fecha(str(x))).all():
                st.session_state.df[columna] = pd.to_datetime(st.session_state.df[columna])

    if st.button("Ver datos"):
        st.session_state.pagina_actual = "ANALISIS"


# Página de ANALISIS
def page_analisis():
    st.title("ANALISIS EXPLORATORIO DEL DATASET")
    st.markdown("---")
    st.markdown("<h4 id='estructura-del-dataset' style='text-align: left; color: #FF5733;'>ESTRUCTURA DEL DATASET</h4>", 
    unsafe_allow_html=True)
    st.table(st.session_state.df.head())

    # Conteo del número de columnas y registros
    num_columnas = st.session_state.df.shape[1]
    num_registros = st.session_state.df.shape[0]

    # Validar columnas con valores nulos 
    columnas_con_nulos = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()

    st.markdown("---")  # Separador de secciones
    # Mostrar los resultados en una sola sección 
    st.markdown("<h4 id='resumen-del-dataset' style='text-align: left; color: #FF5733;'>RESUMEN DEL DATASET</h4>", unsafe_allow_html=True)

    # Crear tres columnas para mostrar los resultados
    col1, col2, col3 = st.columns(3)

    # Conteo de columnas en la primera columna
    with col1:
        st.write("Número de Columnas")
        st.write(num_columnas)

    # Conteo de registros en la segunda columna
    with col2:
        st.write("Número de Registros")
        st.write(num_registros)

    # Mostrar las columnas con valores nulos
    with col3:
        st.write("Columnas con Valores Nulos")
        for columna in columnas_con_nulos:
            st.text(columna)

    # Seccion tipos de Varibales
    st.markdown("---")  # Separador de secciones
    st.markdown("<h4 id='tipos-de-variables' style='text-align: left; color: #FF5733;'>TIPOS DE VARIABLES</h4>", 
    unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("Numéricas continuas")
        num_continuas = [column for column in st.session_state.df.columns 
        if st.session_state.df[column].dtype in ['float64', 'int64'] and st.session_state.df[column].nunique() > 20]
        for var in num_continuas:
            st.text(var)

    with col2:
        st.write("Numéricas discretas")
        num_discretas = [column for column in st.session_state.df.columns 
        if st.session_state.df[column].dtype in ['int64'] and st.session_state.df[column].nunique() <= 20]
        for var in num_discretas:
            st.text(var)

    with col3:
        st.write("Categóricas")
        categoricas = [column for column in st.session_state.df.columns 
        if st.session_state.df[column].dtype == 'object']
        for var in categoricas:
            st.text(var)

    with col4:
        st.write("Fecha")
        fechas = [column for column in st.session_state.df.columns 
        if st.session_state.df[column].dtype == 'datetime64[ns]']
        for var in fechas:
            st.text(var)


    # Seccion tipos de Varibales
    st.markdown("---")  # Separador de secciones
    st.markdown("<h4 id='graficos' style='text-align: left; color: #FF5733;'>ANALISIS DESCRIPTIVO DE LAS VARIABLES</h4>", 
    unsafe_allow_html=True)
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>VARIABLES NUMERICAS CONTINUAS</h6>", 
    unsafe_allow_html=True)

    # Seleccionar una variable numérica continua
    selected_variable = st.selectbox("Selecciona una variable:", num_continuas)
    
    # Calcular estadísticas
    media = st.session_state.df[selected_variable].mean()
    mediana = st.session_state.df[selected_variable].median()
    desviacion_estandar = st.session_state.df[selected_variable].std()
    varianza = st.session_state.df[selected_variable].var()

    # Crear cuatro columnas para mostrar las estadísticas
    col1, col2, col3, col4 = st.columns(4)

    # Mostrar las estadísticas en las columnas correspondientes
    with col1:
        st.write("Media")
        st.write(media)

    with col2:
        st.write("Mediana")
        st.write(mediana)

    with col3:
        st.write("Desviación Estándar")
        st.write(desviacion_estandar)

    with col4:
        st.write("Varianza")
        st.write(varianza)

    # Crear el gráfico de densidad con histograma
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(st.session_state.df[selected_variable], kde=True, color="steelblue", ax=ax)
    ax.set_title(f'Distribución de {selected_variable}')

    # Mostrar el gráfico
    mostrar_matplotlib_fig(fig)

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>VARIABLES NUMERICAS DISCRETAS</h6>", 
    unsafe_allow_html=True)

    # Seleccionar una variable numérica continua
    selected_variable = st.selectbox("Selecciona una variable:", num_discretas)
    
    # Calcular estadísticas
    media = st.session_state.df[selected_variable].mean()
    mediana = st.session_state.df[selected_variable].median()
    desviacion_estandar = st.session_state.df[selected_variable].std()
    moda = st.session_state.df[selected_variable].mode().values[0]

    # Crear cuatro columnas para mostrar las estadísticas
    col1, col2, col3, col4 = st.columns(4)

    # Mostrar las estadísticas en las columnas correspondientes
    with col1:
        st.write("Media")
        st.write(media)

    with col2:
        st.write("Mediana")
        st.write(mediana)

    with col3:
        st.write("Desviación Estándar")
        st.write(desviacion_estandar)

    with col4:
        st.write("Moda")
        st.write(moda)

    ## Crear el gráfico de histograma
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(st.session_state.df[selected_variable], color="steelblue", ax=ax)
    ax.set_title(f'Distribución de {selected_variable}')

    # Mostrar el gráfico
    mostrar_matplotlib_fig(fig)

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>VARIABLES CATEGORICAS</h6>", 
    unsafe_allow_html=True)

    if not categoricas:
        st.warning("SIN DATOS PARA MOSTRAR EN ESTE GRAFICO")
    else:
        # Selecciona una variable categórica
        selected_variable = st.selectbox("Selecciona una variable categórica:", categoricas)

        # Cuenta los valores totales de cada categoría
        counts = st.session_state.df[selected_variable].value_counts()

        if counts.empty:
            st.warning("SIN DATOS PARA MOSTRAR EN ESTE GRAFICO")
        else:
            # Crea el gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=counts.index, y=counts.values, palette="Set3", ax=ax)
            ax.set_title(f'Conteo de valores en la variable {selected_variable}')
            ax.set_xlabel(selected_variable)
            ax.set_ylabel("Conteo")

            # Muestra el gráfico
            st.pyplot(fig)

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>GRAFICO COMBINADO VARIABLES NUMERICAS</h6>", 
    unsafe_allow_html=True)

    # Seleccionar variables para los ejes X e Y
    st.markdown("Selecciona las variables para los ejes X e Y:")

    # Organizar los select boxes en dos columnas
    col1, col2 = st.columns(2)
    
    variable_x = col1.selectbox("Eje X", num_continuas + num_discretas)
    variable_y = col2.selectbox("Eje Y", num_continuas + num_discretas)

    # Crear el gráfico de dispersión
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(st.session_state.df[variable_x], st.session_state.df[variable_y])
    ax.set_xlabel(variable_x)
    ax.set_ylabel(variable_y)
    ax.set_title(f'Gráfico de Dispersión de {variable_x} vs. {variable_y}')
    st.pyplot(fig)

    # Calcular la correlación
    correlacion = st.session_state.df[variable_x].corr(st.session_state.df[variable_y])
    st.write(f"Correlación entre {variable_x} y {variable_y}: {correlacion:.2f}")

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>GRAFICO DE SERIE TEMPORAL</h6>", 
    unsafe_allow_html=True)

    # Seleccionar variables para los ejes X e Y
    st.markdown("Selecciona las variables para los ejes X e Y:")

    # Organizar los select boxes en dos columnas
    col1, col2 = st.columns(2)

    if not fechas:
        st.warning("SIN DATOS PARA MOSTRAR EN ESTE GRAFICO")
    else:
        variable_x = col1.selectbox("Eje X", fechas, key="selectbox_x")
        variable_y = col2.selectbox("Eje Y", num_continuas + num_discretas, key="selectbox_y")


        # Crear el gráfico de serie temporal
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.session_state.df[variable_x], st.session_state.df[variable_y])
        ax.set_xlabel(variable_x)
        ax.set_ylabel(variable_y)
        ax.set_title(f'Gráfico de Serie Temporal de {variable_y} a lo largo del tiempo')
        st.pyplot(fig)

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>GRAFICO DE VARIABLES CATEGORICAS VS VARIABLES CONTINUAS</h6>", 
    unsafe_allow_html=True)

    # Seleccionar variables para los ejes X e Y
    st.markdown("Selecciona las variables para los ejes X e Y:")

    # Organizar los select boxes en dos columnas
    col1, col2 = st.columns(2)

    # Validar si hay variables categóricas y numéricas continuas
    if not categoricas or not num_continuas:
        st.error("SIN DATOS PARA MOSTRAR EN ESTE GRAFICO")
    else:
        variable_x = col1.selectbox("Eje X (Categórica)", categoricas)
        variable_y = col2.selectbox("Eje Y (Numérica Continua)", num_continuas)

        # Crear el gráfico de boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=st.session_state.df[variable_x], y=st.session_state.df[variable_y], ax=ax)
        ax.set_xlabel(variable_x)
        ax.set_ylabel(variable_y)
        ax.set_title(f'Boxplot de {variable_y} según {variable_x}')
        st.pyplot(fig)

    # Salto en la misma sección
    st.write("")
    st.markdown("<h6 style='text-align: left; color: #2145a8;'>GRAFICO DE MOSAICO ENTRE VARIABLES CATEGORICAS</h6>", 
    unsafe_allow_html=True)

    # Seleccionar variables para los ejes X e Y
    st.markdown("Selecciona las variables para los ejes X e Y:")

    col1, col2 = st.columns(2)

    # Validar si hay variables categóricas
    if not categoricas:
        st.error("SIN DATOS PARA MOSTRAR EN ESTE GRAFICO")
    else:
        variable_x = col1.selectbox("Eje X (Categórica)", categoricas, key="selectbox_x")
        variable_y = col2.selectbox("Eje Y (Categórica)", categoricas, key="selectbox_y")

        if variable_x == variable_y:
            st.error("SELECCIONAR DIFERENTES VARIABLES PARA GENERAR UN GRAFICO")
        else:
            # Crear el gráfico de rectángulos (Treemap)
            fig = px.treemap(st.session_state.df,
                            path=[variable_y, variable_x])

            # Actualizar el diseño para centrar el título
            fig.update_layout(title_text=f'Gráfico de Rectángulos entre {variable_x} y {variable_y}',
                  title_x=0.5,
                  title_xanchor='center')


            # Muestra el gráfico
            st.plotly_chart(fig)


# Inicialización de la página actual
if "pagina_actual" not in st.session_state:
    st.session_state.pagina_actual = "INICIO"

# Crear un menú lateral con botones para cambiar entre páginas
st.sidebar.header("Menu de Navegación")
st.sidebar.markdown("")
if st.sidebar.button("INICIO"):
    st.session_state.pagina_actual = "INICIO"

# Agregar una división en el Sidebar
st.sidebar.markdown("---")

if st.sidebar.button("ANALISIS"):
    st.session_state.pagina_actual = "ANALISIS"

    # Subsecciones de ANALISIS
    subsecciones_analisis = ["ESTRUCTURA DEL DATASET", "RESUMEN DEL DATASET", "TIPOS DE VARIABLES", "ANALISIS DESCRIPTIVO DE LAS VARIABLES"]

    # Agregar enlaces a las subsecciones en el Sidebar
    st.sidebar.markdown("<h5>Subsecciones:</h5>", unsafe_allow_html=True)
    for subseccion in subsecciones_analisis:
        st.sidebar.markdown(f"- [{subseccion}](#{subseccion.replace(' ', '-').lower()})", unsafe_allow_html=True)


# Agregar una división en el Sidebar
st.sidebar.markdown("---")

# Agregar texto con el logo de Streamlit
st.sidebar.markdown(
        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by Jonathan Maldonado</h6>',
        unsafe_allow_html=True,
    )

# Mostrar la página actual
if st.session_state.pagina_actual == "INICIO":
    page_inicio()
elif st.session_state.pagina_actual == "ANALISIS":
    page_analisis()

    

