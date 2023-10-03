import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación (sigmoid)
def derivada_sigmoid(x):
    return x * (1 - x)

# Lectura de datos desde el archivo CSV
datos = pd.read_csv('concentlite.csv')
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values.reshape(-1, 1)

# Normalización de datos (escalar entre 0 y 1)
X = X / np.amax(X, axis=0)
y = y / 100

# Solicitar al usuario la cantidad de capas y neuronas por capa
cantidad_capas = int(input("Ingrese la cantidad de capas en la red neuronal: "))
neuronas_por_capa = []
for i in range(cantidad_capas):
    neuronas = int(input(f"Ingrese la cantidad de neuronas para la capa {i + 1}: "))
    neuronas_por_capa.append(neuronas)

# Definición de la arquitectura de la red neuronal
neuronas_entrada = X.shape[1]
neuronas_salida = 1

# Inicialización aleatoria de los pesos de la red
pesos = []
neuronas_previas = neuronas_entrada
for neuronas in neuronas_por_capa:
    pesos.append(np.random.uniform(size=(neuronas_previas, neuronas)))
    neuronas_previas = neuronas
pesos.append(np.random.uniform(size=(neuronas_previas, neuronas_salida)))

# Hiperparámetros
tasa_aprendizaje = 0.1
epsilon = 1e-8
epocas = 10000

# Entrenamiento de la red neuronal con RMSprop
for epoca in range(epocas):
    # Propagación hacia adelante
    capas = [X]
    for i in range(cantidad_capas + 1):
        entrada = np.dot(capas[i], pesos[i])
        salida = sigmoid(entrada)
        capas.append(salida)
    
    # Cálculo del error
    error = y - capas[-1]
    
    # Retropropagación y ajuste de pesos con RMSprop
    deltas = [error * derivada_sigmoid(capas[-1])]
    for i in range(cantidad_capas, 0, -1):
        delta = deltas[-1].dot(pesos[i].T) * derivada_sigmoid(capas[i])
        deltas.append(delta)
    deltas.reverse()
    
    # Actualización de pesos con RMSprop
    for i in range(cantidad_capas + 1):
        media_cuadratica = np.mean(np.square(deltas[i]), axis=0)
        pesos[i] += (capas[i].T.dot(deltas[i]) / (np.sqrt(media_cuadratica) + epsilon)) * tasa_aprendizaje

# Clasificación de los datos
capas = [X]
for i in range(cantidad_capas + 1):
    entrada = np.dot(capas[i], pesos[i])
    salida = sigmoid(entrada)
    capas.append(salida)
salida_predicha = capas[-1]

# Visualización de la distribución de clases
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Distribución de Clases para el Dataset con RMSprop')
plt.colorbar(label='Clase')
plt.show()
