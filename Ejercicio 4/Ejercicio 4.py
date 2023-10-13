import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos desde irisbin.csv
datos = pd.read_csv('irisbin.csv', header=None)

# Extraer características (primeras 4 columnas) y etiquetas (últimas 3 columnas)
X = datos.iloc[:, :4].values
y = datos.iloc[:, 4:].values

# Convertir las etiquetas codificadas en one-hot a su equivalente ordinal
y = np.argmax(y, axis=1)

# Escalar las características para asegurarse de que estén en la misma escala
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.2, random_state=42)

# Crear el modelo MLP (Perceptrón Multicapa)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
mlp.fit(X_entrenamiento, y_entrenamiento)

# Hacer predicciones en el conjunto de prueba
predicciones = mlp.predict(X_prueba)

# Calcular la precisión del modelo
precision = accuracy_score(y_prueba, predicciones)
print("Precisión del modelo: {:.2f}%".format(precision * 100))

# Reducir la dimensionalidad a 2D para visualización usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

# Graficar la proyección 2D de las clases
plt.figure(figsize=(8, 6))
for i in range(3):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1])

plt.title('Proyección en dos dimensiones de la distribución de clases para el conjunto de datos Iris')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
