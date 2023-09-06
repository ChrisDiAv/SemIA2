import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Función de activación 
def funcion_escalon(x):
    return 1 if x >= 0 else -1

#Entrenar el perceptrón
def entrenar_perceptron(X, y, tasa_aprendizaje, max_epocas):
    num_entradas = X.shape[1]
    num_muestras = X.shape[0]
    
    pesos = np.random.rand(num_entradas)
    sesgo = np.random.rand()
    
    for epoca in range(max_epocas):
        error = 0
        for i in range(num_muestras):
            entrada_neta = np.dot(X[i], pesos) + sesgo
            salida = funcion_escalon(entrada_neta)
            
            
            delta = y[i] - salida
            
            pesos += tasa_aprendizaje * delta * X[i]
            sesgo += tasa_aprendizaje * delta
            
            error += delta ** 2
        
        error /= 2
        print(f"Época {epoca + 1}/{max_epocas}, Error = {error}")
        
        
        if error == 0:
            break
    
    return pesos, sesgo

def probar_perceptron(X, pesos, sesgo):
    num_muestras = X.shape[0]
    predicciones = []
    
    for i in range(num_muestras):
        entrada_neta = np.dot(X[i], pesos) + sesgo
        salida = funcion_escalon(entrada_neta)
        predicciones.append(salida)
    
    return predicciones

# Lectura de datos de entrenamiento
X_entrenamiento = np.genfromtxt('XOR_trn.csv', delimiter=',')
y_entrenamiento = np.genfromtxt('XOR_trn.csv', delimiter=',', usecols=-1)

X_prueba = np.genfromtxt('XOR_tst.csv', delimiter=',')
y_prueba = np.genfromtxt('XOR_tst.csv', delimiter=',', usecols=-1)

# Entrenamiento del perceptrón
tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje: "))
max_epocas = int(input("Ingrese el número máximo de épocas de entrenamiento: "))

pesos, sesgo = entrenar_perceptron(X_entrenamiento, y_entrenamiento, tasa_aprendizaje, max_epocas)

# Prueba del perceptrón
predicciones = probar_perceptron(X_prueba, pesos, sesgo)

# Mostrar resultados y gráfica
print("Predicciones en el conjunto de prueba:")
for i, prediccion in enumerate(predicciones):
    print(f"Patrón {i+1}: {prediccion}")

# Gráfico de los patrones 
plt.scatter(X_entrenamiento[y_entrenamiento == 1][:, 0], X_entrenamiento[y_entrenamiento == 1][:, 1], marker='o', label='Clase 1', c='b')
plt.scatter(X_entrenamiento[y_entrenamiento == -1][:, 0], X_entrenamiento[y_entrenamiento == -1][:, 1], marker='x', label='Clase -1', c='r')

x_min, x_max = X_entrenamiento[:, 0].min() - 1, X_entrenamiento[:, 0].max() + 1
y_min, y_max = X_entrenamiento[:, 1].min() - 1, X_entrenamiento[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], pesos[:2]) + sesgo
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='g')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.title('Separación de Clases')
plt.show()
