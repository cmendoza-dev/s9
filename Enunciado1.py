import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Cargar el archivo Excel
file_path = './BI_Postulantes09.xlsx'
df = pd.read_excel(file_path)

# Mostrar los nombres exactos de las columnas para verificar
print(df.columns)

# Seleccionar las características relevantes
features = ['Apertura Nuevos Conoc.', 'Nivel Organización', 'Participación Grupo Social', 
            'Grado Empatía', 'Grado Nerviosismo', 'Dependencia Internet']

# Manejar valores nulos (si es necesario)
df = df.dropna(subset=features)

# Asegurarse de que las columnas son numéricas
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Eliminar filas con valores no convertibles
df = df.dropna(subset=features)

# Escalar las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Aplicar k-means
n_clusters = 4  # Puedes ajustar este número según sea necesario
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Verificar tipos de datos de las columnas antes de agrupar
print(df.dtypes)

# Asegurarse de que las columnas usadas para calcular la media son numéricas
numerical_cols = df[features].select_dtypes(include=['number']).columns.tolist()

# Función para generar histogramas cruzando dimensiones
def plot_histograms(df, feature, hue):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=feature, hue=hue, multiple="stack", palette="Set1", kde=False)
    plt.title(f'Histograma de {feature} por {hue}')
    plt.xlabel(feature)
    plt.ylabel('Cuenta')
    # Agregar leyenda personalizada
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                  markerfacecolor=color, markersize=10) 
                       for label, color in zip(df[hue].unique(), sns.color_palette("Set1", n_colors=len(df[hue].unique())))]
    plt.legend(handles=legend_elements, title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

# Crear la aplicación Streamlit
st.title('Análisis de Conglomerados de Postulantes')

# Mostrar información general
st.write("### Información General del Dataset")
st.write(df.describe())

# Seleccionar característica para el histograma
selected_feature = st.selectbox('Selecciona una característica:', features)

# Mostrar histograma para la característica seleccionada
plot_histograms(df, selected_feature, 'Nom_Especialidad')

# Mostrar tabla de correspondencia entre colores y especialidades
st.subheader('Correspondencia de Colores y Especialidades')
color_specialty_mapping = df[['cluster', 'Nom_Especialidad']].drop_duplicates().set_index('cluster')
st.table(color_specialty_mapping)

# Mostrar información de los clusters
st.subheader('Información de Clusters')
cluster_info = df.groupby('cluster')[numerical_cols].mean()
st.write(cluster_info)

# Mostrar asignaciones de clusters
st.subheader('Asignaciones de Clusters')
st.write(df[['Nom_Especialidad', 'cluster']])