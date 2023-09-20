#Trabajo en equipo 
#Alan Gonzalez Olmos - 219748286
#Christian Jhareth Diaz AVila - 215662344

import numpy as np
import matplotlib.pyplot as plt

# Definir la función de pérdida
def funcion_perdida(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Definir el gradiente de la función de pérdida
def gradiente(x1, x2):
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3*x2**2))
    return np.array([df_dx1, df_dx2])

# Inicializar los parámetros
x1 = 1.0
x2 = 1.0
tasa_aprendizaje = 0.1
num_iteraciones = 100

# Listas para almacenar datos de seguimiento
historial_x1 = [x1]
historial_x2 = [x2]
historial_perdida = [funcion_perdida(x1, x2)]

# Descenso del gradiente
for i in range(num_iteraciones):
    grad = gradiente(x1, x2)
    # Actualizar x1 y x2 utilizando la tasa de aprendizaje y el gradiente
    x1 -= tasa_aprendizaje * grad[0]
    x2 -= tasa_aprendizaje * grad[1]
    # Calcular la pérdida en el nuevo punto
    perdida = funcion_perdida(x1, x2)
    # Registrar los valores actuales en el historial
    historial_x1.append(x1)
    historial_x2.append(x2)
    historial_perdida.append(perdida)
    # Imprimir información de seguimiento en cada iteración
    print(f"Iteración {i+1}: x1 = {x1}, x2 = {x2}, Pérdida = {perdida}")

print("Resultado final:")
print(f"x1 = {x1}, x2 = {x2}, Pérdida final = {perdida}")

# Gráfica de la convergencia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial_x1, label='x1')
plt.plot(historial_x2, label='x2')
plt.xlabel('Iteración')
plt.ylabel('Valor')
plt.legend()
plt.title('Convergencia de x1 y x2')

plt.subplot(1, 2, 2)
plt.plot(historial_perdida, label='Pérdida')
plt.xlabel('Iteración')
plt.ylabel('Valor de Pérdida')
plt.legend()
plt.title('Convergencia de la Pérdida')

plt.tight_layout()
plt.show()
