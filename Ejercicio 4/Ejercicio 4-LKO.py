import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos desde 'irisbin.csv'
datos = pd.read_csv('irisbin.csv', header=None)

# Extraer características (primeras 4 columnas) y etiquetas (últimas 3 columnas)
X = datos.iloc[:, :4].values
y = np.argmax(datos.iloc[:, 4:].values, axis=1)

# Convertir las etiquetas codificadas en one-hot a su forma ordinal
# Estandarizar las características para asegurar que estén en la misma escala
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Crear el modelo de MLP (Perceptrón Multicapa)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Generar un número aleatorio entre 1 y 10 para k
k = np.random.randint(1, 11)

# Utilizar el método de validación cruzada leave-k-out para evaluar el modelo
puntuaciones = cross_val_score(mlp, X_escalado, y, cv=k)

# Calcular el error esperado de clasificación, la puntuación promedio y la desviación estándar
error_clasificacion = 1 - puntuaciones.mean()
puntuacion_promedio = puntuaciones.mean()
desviacion_estandar = puntuaciones.std()

print("Resultados de la Validación Cruzada Leave-{}-Out:".format(k))
print("Error de Clasificación: {:.2f}".format(error_clasificacion))
print("Puntuación Promedio: {:.2f}".format(puntuacion_promedio))
print("Desviación Estándar: {:.2f}".format(desviacion_estandar))

# Reducir la dimensionalidad a 2D para la visualización utilizando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

# Graficar la proyección en 2D de las clases
plt.figure(figsize=(8, 6))
for i in range(3):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Clase {i}')
    
plt.title('Proyección en dos dimensiones de la distribución de clases para el conjunto de datos Iris')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
