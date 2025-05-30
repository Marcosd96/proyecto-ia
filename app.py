import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Cargar y preparar los datos igual que en modelo.py
csv_path = "StudentPerformanceFactors.csv"
df = pd.read_csv(csv_path)

def clasificar_puntaje(score):
    if score < 60:
        return "Bajo"
    elif score < 80:
        return "Medio"
    else:
        return "Alto"

df["Rendimiento"] = df["Exam_Score"].apply(clasificar_puntaje)
df.drop(columns=["Exam_Score"], inplace=True)

# Guardar los nombres de las columnas y los labelencoders para usarlos en la predicción
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop(columns=["Rendimiento"])
y = df["Rendimiento"]

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)

# Evaluar el modelo en el conjunto de prueba
score = rf.score(X_test_scaled, y_test)
print(f"Precisión en el conjunto de prueba: {score:.2%}")

# Interfaz gráfica
campos = list(X.columns)

root = tk.Tk()
root.title("Predicción de Rendimiento Estudiantil")

# Definición de opciones para campos categóricos
opciones = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["Yes", "No"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Positive", "Neutral", "Negative"],
    "Learning_Disabilities": ["Yes", "No"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"]
}

entradas = {}
for i, campo in enumerate(campos):
    tk.Label(root, text=campo).grid(row=i, column=0, sticky="w")
    if campo in opciones:
        entradas[campo] = ttk.Combobox(root, values=opciones[campo], state="readonly")
        entradas[campo].current(0)
    else:
        entradas[campo] = tk.Entry(root)
    entradas[campo].grid(row=i, column=1)

resultado_var = tk.StringVar()
resultado_label = tk.Label(root, textvariable=resultado_var, font=("Arial", 14, "bold"))
resultado_label.grid(row=len(campos), column=0, columnspan=2, pady=10)

def predecir():
    datos = []
    for campo in campos:
        if campo in opciones:
            valor = entradas[campo].get()
        else:
            valor = entradas[campo].get()
            try:
                valor = float(valor)
            except ValueError:
                pass
        datos.append(valor)
    # Crear DataFrame para un solo registro
    df_nuevo = pd.DataFrame([datos], columns=campos)
    # Codificar igual que el entrenamiento
    for col in df_nuevo.columns:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                df_nuevo[col] = le.transform([str(df_nuevo[col][0])])
            except ValueError:
                messagebox.showerror("Error", f"Valor no válido para {col}. Opciones: {list(le.classes_)}")
                return
    # Escalar
    X_nuevo = scaler.transform(df_nuevo)
    pred = rf.predict(X_nuevo)[0]
    # Decodificar resultado
    if hasattr(label_encoders["Rendimiento"], "inverse_transform"):
        resultado = label_encoders["Rendimiento"].inverse_transform([pred])[0]
    else:
        resultado = pred
    resultado_var.set(f"Predicción: {resultado}")

btn = tk.Button(root, text="Predecir", command=predecir)
btn.grid(row=len(campos)+1, column=0, columnspan=2, pady=10)

root.mainloop()
