# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Carga el conjunto de datos
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["embarazos", "glucosa", "presion_sanguinea", "grosor_piel", "insulina", "IMC", "pedigree_diabetes", "edad", "resultado"]
data = pd.read_csv(url, names=column_names)

# Define las características (X) y las etiquetas (y)
X = data.drop("resultado", axis=1)
y = data["resultado"]

# Divide el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para mejorar el rendimiento del modelo (NB no requiere normalización, pero puede ayudar en algunos casos)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea y entrena el modelo de Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = naive_bayes.predict(X_test)

# Calcula la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)