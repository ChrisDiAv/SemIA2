import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
datos = pd.read_csv("spheres1d10.csv") #Tabla1.csv

# Dividir los datos en características y objetivo
X = datos.iloc[:, :-1]  
y = datos.iloc[:, -1]   

# Inicializar el modelo
modelo = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for i in range(5):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=i)

    # Entrenar el modelo
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Evaluar el modelo
    exactitud = modelo.score(X_prueba, y_prueba)
    ax1.plot(i, exactitud, marker='o', linestyle='--', label=f'Partición {i}')

# Gráfica 1: Precisión del modelo para cada partición
ax1.set_title('Exactitud del Modelo para Cada Partición')
ax1.set_xlabel('Número de Partición')
ax1.set_ylabel('Exactitud')
ax1.grid()
ax1.legend()

# Gráfica 2: Real vs. Predicho
y_predicho = modelo.predict(X_prueba)
ax2.scatter(y_prueba, y_predicho)
ax2.set_xlabel('Real')
ax2.set_ylabel('Predicho')
ax2.set_title('Real vs Predicho')
plt.tight_layout()
plt.show()