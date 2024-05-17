import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import streamlit as st

# Cargar el archivo Excel
file_path = './BI_Clientes09.xlsx'
df = pd.read_excel(file_path)

# Seleccionar una variable como objetivo del árbol de decisiones
target_variable = 'BikeBuyer'

# Seleccionar características para el árbol de decisiones
features = ['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'NumberCarsOwned', 'Age']

# Eliminar filas con valores nulos
df = df.dropna(subset=[target_variable] + features)

# Separar las características y la variable objetivo
X = df[features]
y = df[target_variable]

# Crear el modelo de árbol de decisiones
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Visualizar el árbol de decisiones
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=['No Comprador', 'Comprador'], filled=True)
plt.title('Árbol de Decisiones para Predicción de Compradores de Bicicletas')
st.pyplot(plt)
