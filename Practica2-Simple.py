import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
data = pd.read_csv('spheres1d10.csv') #Tabla1.csv

# Dividir los datos en características y objetivo
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

# Inicializar el modelo
modelo = Perceptron()

precisiones = []
for i in range(5):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=i)

    # Entrenar el modelo
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Evaluar el modelo
    precisión = modelo.score(X_prueba, y_prueba)
    precisiones.append(precisión)
    print("Precisión del modelo para la partición ", i, ": ", precisión)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Gráfica 1: Precisión del modelo para cada partición
ax1.plot(range(5), precisiones, marker='o', linestyle='--')
ax1.set_title('Precisión del Modelo para Cada Partición')
ax1.set_xlabel('Número de Partición')
ax1.set_ylabel('Precisión')
ax1.grid()

# Gráfica 2: Real vs. Predicho
y_predicho = modelo.predict(X_prueba)
sns.scatterplot(x=y_prueba, y=y_predicho, ax=ax2)
ax2.set_xlabel('Real')
ax2.set_ylabel('Predicho')
ax2.set_title('Real vs Predicho')
plt.tight_layout()
plt.show()