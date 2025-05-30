###UTILIZACION DE DECISION TREE Y RANDOM FOREST#######

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 1. Cargar y preparar los datos
df = pd.read_csv("StudentPerformanceFactors.csv")

def clasificar_puntaje(score):
    if score < 60:
        return "Bajo"
    elif score < 80:
        return "Medio"
    else:
        return "Alto"

df["Rendimiento"] = df["Exam_Score"].apply(clasificar_puntaje)
df.drop(columns=["Exam_Score"], inplace=True)

# 2. Codificar variables categóricas
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 3. Separar características y etiquetas
X = df.drop(columns=["Rendimiento"])
y = df["Rendimiento"]

# Escalar (opcional para árboles, pero mantenemos consistencia)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Aplicar SMOTE para balancear
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 5. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# === DECISION TREE ===
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("=== Árbol de Decisión ===")
print(classification_report(y_test, y_pred_tree))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree, display_labels=["Bajo", "Medio", "Alto"])
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.tight_layout()
plt.show()

# Visualizar árbol (opcional si es muy grande)
plt.figure(figsize=(20, 8))
plot_tree(tree, feature_names=X.columns, class_names=["Bajo", "Medio", "Alto"], filled=True, max_depth=3)
plt.title("Árbol de Decisión (primeras ramas)")
plt.show()

# === RANDOM FOREST ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=["Bajo", "Medio", "Alto"])
plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()
plt.show()

# Importancia de características
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=importancias, y=importancias.index)
plt.title("Importancia de características - Random Forest")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

import imblearn
print(imblearn.__path__)