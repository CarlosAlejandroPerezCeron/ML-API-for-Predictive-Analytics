import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos de ejemplo
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo entrenado
joblib.dump(model, "model/model.pkl")
print("✅ Modelo guardado en model/model.pkl")
