import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos desde irisbin.csv
datos = pd.read_csv('irisbin.csv', header=None)

# Extraer características (primeras 4 columnas) y etiquetas (últimas 3 columnas)
X = datos.iloc[:, :4].values
y = datos.iloc[:, 4:].values
y = np.argmax(y, axis=1)

# Escalar características para asegurar que estén en la misma escala
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Configurar el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Configurar el método de validación Leave-One-Out
loo = LeaveOneOut()

# Realizar validación cruzada Leave-One-Out
resultados_loo = cross_val_score(mlp, X_escalado, y, cv=loo)

# Calcular el error esperado de clasificación
error_clasificacion = 1 - resultados_loo.mean()

print("Error esperado de clasificación: {:.2f}%".format(error_clasificacion * 100))

# Calcular el promedio y la desviación estándar de ambos métodos
promedio_loo = resultados_loo.mean()
desviacion_estandar_loo = resultados_loo.std()

print("Promedio LOO: {:.2f}%".format(promedio_loo * 100))
print("Desviación estándar LOO: {:.2f}".format(desviacion_estandar_loo))

# Reducir la dimensionalidad a 2D para visualización usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

# Graficar la proyección 2D de las clases
plt.figure(figsize=(8, 6))
for i in range(3):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1])
    
plt.title('Proyección en dos dimensiones de la distribución de clases para el dataset Iris')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
