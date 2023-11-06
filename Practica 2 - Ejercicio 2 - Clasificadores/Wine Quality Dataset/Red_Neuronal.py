import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar el dataset desde la URL proporcionada
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=';')

# Separar características (X) y etiquetas (y)
X = data.drop('quality', axis=1)  # X contiene las características del vino
y = data['quality']  # y contiene las etiquetas de calidad del vino

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características para normalizarlas
scaler = StandardScaler()  # Crea un objeto para escalar las características
X_train = scaler.fit_transform(X_train)  # Escala las características de entrenamiento
X_test = scaler.transform(X_test)  # Escala las características de prueba usando la misma escala que se usó en el conjunto de entrenamiento

# Construir el modelo de red neuronal
model = Sequential()  # Crea un modelo de red neuronal secuencial
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Capa oculta con 128 neuronas y función de activación ReLU
model.add(Dense(64, activation='relu'))  # Otra capa oculta con 64 neuronas y función de activación ReLU
model.add(Dense(1, activation='linear'))  # Capa de salida con 1 neurona para problemas de regresión y función de activación lineal

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam')  # Usamos el error cuadrático medio como función de pérdida y el optimizador Adam

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)  # Entrenamos el modelo con los datos de entrenamiento, usando lotes de 16 muestras por época y validamos con el 10% de los datos de entrenamiento

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test)  # Calculamos la pérdida en el conjunto de prueba
print(f"Pérdida en el conjunto de prueba: {loss}")  # Mostramos la pérdida en el conjunto de prueba

# Predecir etiquetas para el conjunto de prueba
predictions = model.predict(X_test)  # Usamos el modelo entrenado para predecir etiquetas para el conjunto de prueba
