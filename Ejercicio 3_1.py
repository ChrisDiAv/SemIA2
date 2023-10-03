import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación (sigmoide)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación
def derivada_sigmoide(x):
    return x * (1 - x)

# Lectura de datos desde el archivo CSV
datos = pd.read_csv("concentlite.csv")
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values.reshape(-1, 1)

# Normalización de datos (escalar entre 0 y 1)
X = X / np.amax(X, axis=0)
y = y / 100

# Obtener la arquitectura de la red desde la entrada del usuario
num_capas = int(input("Ingrese el número de capas en la red: "))
neuronas_por_capa = []
for i in range(num_capas):
    neuronas = int(input(f"Ingrese el número de neuronas en la capa {i + 1}: "))
    neuronas_por_capa.append(neuronas)

# Definición de la arquitectura de la red neuronal
neuronas_entrada = X.shape[1]
neuronas_salida = 1

# Inicialización aleatoria de los pesos de la red
pesos = []
for i in range(num_capas):
    if i == 0:
        pesos.append(np.random.uniform(size=(neuronas_entrada, neuronas_por_capa[i])))
    else:
        pesos.append(np.random.uniform(size=(neuronas_por_capa[i - 1], neuronas_por_capa[i])))
pesos.append(np.random.uniform(size=(neuronas_por_capa[-1], neuronas_salida)))

# Hiperparámetros
tasa_aprendizaje = 0.1
epocas = 10000

# Entrenamiento de la red neuronal
for epoca in range(epocas):
    # Propagación hacia adelante
    salidas_capas = []
    entradas_capas = []
    for i in range(num_capas + 1):
        if i == 0:
            entrada_capa = np.dot(X, pesos[i])
        else:
            entrada_capa = np.dot(salidas_capas[i - 1], pesos[i])
        entradas_capas.append(entrada_capa)
        salida_capa = sigmoide(entrada_capa)
        salidas_capas.append(salida_capa)
    
    # Cálculo del error
    error = y - salidas_capas[-1]
    
    # Retropropagación y ajuste de pesos
    deltas = []
    for i in reversed(range(num_capas + 1)):
        if i == num_capas:
            delta = error * derivada_sigmoide(salidas_capas[i])
        else:
            delta = deltas[-1].dot(pesos[i + 1].T) * derivada_sigmoide(salidas_capas[i])
        deltas.append(delta)
    
    deltas.reverse()
    
    for i in range(num_capas + 1):
        if i == 0:
            pesos[i] += X.T.dot(deltas[i]) * tasa_aprendizaje
        else:
            pesos[i] += salidas_capas[i - 1].T.dot(deltas[i]) * tasa_aprendizaje

# Clasificación de los datos
entrada_capa = np.dot(X, pesos[0])
salida_capa = sigmoide(entrada_capa)
for i in range(1, num_capas + 1):
    entrada_capa = np.dot(salida_capa, pesos[i])
    salida_capa = sigmoide(entrada_capa)
salida_predicha = salida_capa

# Visualización de la distribución de clases
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Distribución de Clases para el Conjunto de Datos')
plt.colorbar(label='Clase')
plt.show()
