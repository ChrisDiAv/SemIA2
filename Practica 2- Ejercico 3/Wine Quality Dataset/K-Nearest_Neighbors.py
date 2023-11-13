import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Carga el conjunto de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# Define las características (X) y las etiquetas (y)
X = data.drop("quality", axis=1)  # Características
y = data["quality"]  # Etiquetas

# Divide el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Normaliza los datos de entrenamiento
X_test = scaler.transform(X_test)  # Normaliza los datos de prueba usando la misma escala que los datos de entrenamiento

# Entrena el modelo de k-vecinos cercanos
k = 5  # Número de vecinos a considerar (puedes ajustar este valor según sea necesario)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)  # Entrena el modelo usando los datos de entrenamiento normalizados

# Realiza predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)  # Realiza predicciones usando el modelo entrenado y datos de prueba normalizados

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)  # Calcula la precisión comparando las predicciones con las etiquetas reales
print("Precisión del modelo:", accuracy)  # Imprime la precisión del modelo en el conjunto de prueba

# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcula Precision, Recall (Sensitivity), Specificity y F1 Score
precision = precision_score(y_test, y_pred, average='weighted')  # Usamos 'weighted' para datos multiclase
recall = recall_score(y_test, y_pred, average='weighted')
specificity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
f1 = f1_score(y_test, y_pred, average='weighted')

# Imprime las métricas adicionales
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
