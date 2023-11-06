# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carga el conjunto de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")
# Si estás utilizando el conjunto de datos de vinos blancos, usa la siguiente URL:
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
# data = pd.read_csv(url, sep=";")

# Define las características (X) y las etiquetas (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Divide el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea y entrena el modelo de Máquinas de Soporte Vectorial (SVM)
svm = SVC(kernel='linear', C=1)  # Utiliza un kernel lineal y un parámetro de regularización C=1
svm.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = svm.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)