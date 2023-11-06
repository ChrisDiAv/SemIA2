import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Cargar el conjunto de datos Pima Indians Diabetes Dataset desde URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# Definir los nombres de las columnas para el conjunto de datos
nombres_columnas = ['embarazos', 'glucosa', 'presion_arterial', 'grosor_piel', 'insulina', 'IMC', 'pedigree_diabetes', 'edad', 'resultado']
# Leer el conjunto de datos en un DataFrame de pandas
datos = pd.read_csv(url, names=nombres_columnas)

# Separar características (X) y etiquetas (y)
X = datos.drop('resultado', axis=1) # Características
y = datos['resultado'] # Etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características para normalizarlas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Escalar el conjunto de entrenamiento
X_test = scaler.transform(X_test) # Escalar el conjunto de prueba

# Construir el modelo de red neuronal
modelo = Sequential()
modelo.add(Dense(128, input_dim=X_train.shape[1], activation='relu')) # Capa oculta con 128 neuronas y función de activación ReLU
modelo.add(Dense(64, activation='relu')) # Otra capa oculta con 64 neuronas y función de activación ReLU
modelo.add(Dense(1, activation='sigmoid')) # Capa de salida con 1 neurona y función de activación sigmoide para problemas de clasificación binaria

# Compilar el modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Usamos binary_crossentropy como función de pérdida para problemas de clasificación binaria

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1) # Entrenamos el modelo con 100 épocas y un tamaño de lote de 16, usando el 10% de los datos para validación

# Evaluar el modelo en el conjunto de prueba
pérdida, precisión = modelo.evaluate(X_test, y_test) # Evaluamos el modelo en el conjunto de prueba y obtenemos la pérdida y precisión
print(f"Pérdida en el conjunto de prueba: {pérdida}")
print(f"Precisión en el conjunto de prueba: {precisión}")

# Predecir probabilidades para el conjunto de prueba
probabilidades = modelo.predict(X_test) # Obtenemos las probabilidades de pertenecer a la clase positiva para el conjunto de prueba

# Convertir las probabilidades en clases (0 o 1) usando un umbral de 0.5
umbral = 0.5
predicciones = np.where(probabilidades > umbral, 1, 0) # Convertimos las probabilidades en clases binarias usando el umbral de 0.5