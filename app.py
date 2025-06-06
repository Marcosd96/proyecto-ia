import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargar y preparar los datos
csv_path = "StudentPerformanceFactors.csv"
df = pd.read_csv(csv_path)

def clasificar_puntaje(score):
    if score < 70:
        return "Bajo"
    elif score < 80:
        return "Medio"
    else:
        return "Alto"

df["Rendimiento"] = df["Exam_Score"].apply(clasificar_puntaje)
df.drop(columns=["Exam_Score"], inplace=True)

# --- Diagnóstico: Conteo y muestra de ejemplos 'Bajo' con valores bajos ---
print('--- Diagnóstico: Ejemplos Bajo con valores bajos ---')
bajo_bajos = df[(df['Rendimiento'] == 'Bajo') & (df['Attendance'] <= 60) & (df['Previous_Scores'] <= 51)]
print(f"Total ejemplos 'Bajo' con Attendance <= 60 y Previous_Scores <= 51: {len(bajo_bajos)}")
print(bajo_bajos.head())
# --------------------------------------------------------------------------

def generar_datos_sinteticos_bajo(n=500):
    import numpy as np
    datos = []
    for _ in range(n):
        fila = {
            'Hours_Studied': np.random.randint(1, 6),
            'Attendance': np.random.randint(30, 61),
            'Parental_Involvement': 'Low',
            'Access_to_Resources': 'Low',
            'Extracurricular_Activities': 'No',
            'Sleep_Hours': np.random.randint(3, 6),
            'Previous_Scores': np.random.randint(20, 52),
            'Motivation_Level': 'Low',
            'Internet_Access': 'No',
            'Tutoring_Sessions': 0,
            'Family_Income': 'Low',
            'Teacher_Quality': 'Low',
            'School_Type': 'Public',
            'Peer_Influence': 'Negative',
            'Physical_Activity': 0,
            'Learning_Disabilities': 'Yes',
            'Parental_Education_Level': 'High School',
            'Distance_from_Home': 'Far',
            'Gender': np.random.choice(['Male', 'Female']),
            'Rendimiento': 'Bajo'
        }
        datos.append(fila)
    return pd.DataFrame(datos)

def generar_datos_sinteticos_medio(n=500):
    import numpy as np
    datos = []
    for _ in range(n):
        fila = {
            'Hours_Studied': np.random.randint(6, 15),
            'Attendance': np.random.randint(61, 80),
            'Parental_Involvement': np.random.choice(['Medium', 'Low', 'High']),
            'Access_to_Resources': np.random.choice(['Medium', 'Low', 'High']),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No']),
            'Sleep_Hours': np.random.randint(5, 8),
            'Previous_Scores': np.random.randint(52, 80),
            'Motivation_Level': np.random.choice(['Medium', 'Low', 'High']),
            'Internet_Access': np.random.choice(['Yes', 'No']),
            'Tutoring_Sessions': np.random.randint(0, 3),
            'Family_Income': np.random.choice(['Medium', 'Low', 'High']),
            'Teacher_Quality': np.random.choice(['Medium', 'Low', 'High']),
            'School_Type': np.random.choice(['Public', 'Private']),
            'Peer_Influence': np.random.choice(['Neutral', 'Positive', 'Negative']),
            'Physical_Activity': np.random.randint(1, 4),
            'Learning_Disabilities': np.random.choice(['No', 'Yes']),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate']),
            'Distance_from_Home': np.random.choice(['Moderate', 'Near', 'Far']),
            'Gender': np.random.choice(['Male', 'Female']),
            'Rendimiento': 'Medio'
        }
        datos.append(fila)
    return pd.DataFrame(datos)

def generar_datos_sinteticos_alto(n=500):
    import numpy as np
    datos = []
    for _ in range(n):
        fila = {
            'Hours_Studied': np.random.randint(15, 31),
            'Attendance': np.random.randint(80, 101),
            'Parental_Involvement': 'High',
            'Access_to_Resources': 'High',
            'Extracurricular_Activities': 'Yes',
            'Sleep_Hours': np.random.randint(7, 10),
            'Previous_Scores': np.random.randint(80, 101),
            'Motivation_Level': 'High',
            'Internet_Access': 'Yes',
            'Tutoring_Sessions': np.random.randint(2, 6),
            'Family_Income': 'High',
            'Teacher_Quality': 'High',
            'School_Type': 'Private',
            'Peer_Influence': 'Positive',
            'Physical_Activity': np.random.randint(3, 7),
            'Learning_Disabilities': 'No',
            'Parental_Education_Level': np.random.choice(['College', 'Postgraduate']),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate']),
            'Gender': np.random.choice(['Male', 'Female']),
            'Rendimiento': 'Alto'
        }
        datos.append(fila)
    return pd.DataFrame(datos)

# --- Generar y agregar sintéticos si faltan ejemplos tras el filtrado ---
def asegurar_minimos_clase(df, min_ej=500):
    # Filtrar ejemplos para cada clase
    df_bajo = df[(df['Rendimiento'] == 'Bajo') & (df['Attendance'] <= 60) & (df['Previous_Scores'] <= 51)]
    df_medio = df[(df['Rendimiento'] == 'Medio') & (df['Attendance'] > 60) & (df['Attendance'] < 80) & (df['Previous_Scores'] > 51) & (df['Previous_Scores'] < 80)]
    df_alto = df[(df['Rendimiento'] == 'Alto') & (df['Attendance'] >= 80) & (df['Previous_Scores'] >= 80)]
    faltan_bajo = max(0, min_ej - len(df_bajo))
    faltan_medio = max(0, min_ej - len(df_medio))
    faltan_alto = max(0, min_ej - len(df_alto))
    if faltan_bajo > 0:
        df = pd.concat([df, generar_datos_sinteticos_bajo(faltan_bajo)], ignore_index=True)
    if faltan_medio > 0:
        df = pd.concat([df, generar_datos_sinteticos_medio(faltan_medio)], ignore_index=True)
    if faltan_alto > 0:
        df = pd.concat([df, generar_datos_sinteticos_alto(faltan_alto)], ignore_index=True)
    return df

df = asegurar_minimos_clase(df, min_ej=500)

# Filtrar los ejemplos para que cada clase tenga valores representativos y no solapados
df_bajo_realista = df[(df['Rendimiento'] == 'Bajo') & (df['Attendance'] <= 60) & (df['Previous_Scores'] <= 51)]
df_medio_realista = df[(df['Rendimiento'] == 'Medio') & (df['Attendance'] > 60) & (df['Attendance'] < 80) & (df['Previous_Scores'] > 51) & (df['Previous_Scores'] < 80)]
df_alto_realista = df[(df['Rendimiento'] == 'Alto') & (df['Attendance'] >= 80) & (df['Previous_Scores'] >= 80)]

# Diagnóstico: Conteo de ejemplos por clase tras el filtrado, antes del balanceo
print('\n--- Diagnóstico: Ejemplos por clase tras el filtrado ---')
print(f"Bajo: {len(df_bajo_realista)} | Medio: {len(df_medio_realista)} | Alto: {len(df_alto_realista)}")
if len(df_bajo_realista) < 30 or len(df_medio_realista) < 30 or len(df_alto_realista) < 30:
    print('¡Advertencia! Hay muy pocos ejemplos en alguna clase tras el filtrado. Considera relajar los filtros o agregar datos sintéticos.')

# Balancear el dataset
min_clase = min(len(df_bajo_realista), len(df_medio_realista), len(df_alto_realista))
df_bajo_bal = df_bajo_realista.sample(min_clase, random_state=42)
df_medio_bal = df_medio_realista.sample(min_clase, random_state=42)
df_alto_bal = df_alto_realista.sample(min_clase, random_state=42)
df = pd.concat([df_bajo_bal, df_medio_bal, df_alto_bal], ignore_index=True)

# 2. Codificar variables categóricas (igual que tu código exitoso)
le_rend = LabelEncoder()
df['Rendimiento'] = le_rend.fit_transform(df['Rendimiento'])
for col in df.select_dtypes(include="object").columns:
    if col != 'Rendimiento':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Guardar el mapeo de clases
mapeo_rend = {clase: le_rend.transform([clase])[0] for clase in le_rend.classes_}

# Análisis exploratorio para cualquier clase después del balanceo y codificación
def analizar_clase(clase_nombre):
    if clase_nombre not in mapeo_rend:
        print(f"Clase '{clase_nombre}' no encontrada en el mapeo de clases.")
        return
    clase_cod = mapeo_rend[clase_nombre]
    clase_df = df[df['Rendimiento'] == clase_cod]
    print(f"\n--- Estadísticas para la clase {clase_nombre} (después del balanceo y codificación) ---")
    for var in ['Attendance', 'Hours_Studied', 'Previous_Scores']:
        if var in clase_df.columns:
            print(f"{var}: min={clase_df[var].min()}, max={clase_df[var].max()}, mean={clase_df[var].mean():.2f}, std={clase_df[var].std():.2f}")

# Ejemplo de uso para las tres clases:
analizar_clase('Bajo')
analizar_clase('Medio')
analizar_clase('Alto')

# 3. Separar características y etiquetas
X = df.drop(columns=["Rendimiento"])
y = df["Rendimiento"]

# 4. Escalar (antes de SMOTE, igual que tu código exitoso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Aplicar SMOTE para balancear
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 6. Dividir en entrenamiento y prueba (después de SMOTE)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 7. Entrenar el modelo
# Cambiar a class_weight='balanced' para mejorar sensibilidad a clases minoritarias
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 8. Evaluar el modelo
score = rf.score(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {score:.2%}")
print("\nBalance de clases en el dataset balanceado:")
print(pd.Series(y_resampled).value_counts())
print("\nMatriz de confusión en el conjunto de prueba:")
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Mostrar importancia de variables
campos_importancia = list(X.columns)
importancia = rf.feature_importances_
importancias_ordenadas = sorted(zip(campos_importancia, importancia), key=lambda x: x[1], reverse=True)
print("\nImportancia de variables:")
for nombre, imp in importancias_ordenadas:
    print(f"{nombre}: {imp:.4f}")

# Interfaz gráfica mejorada
campos = list(X.columns)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Predicción de Rendimiento Estudiantil")
    root.geometry("1100x800")  # Tamaño inicial amplio para mostrar toda la interfaz
    root.configure(bg="#f4f6fb")

    # Frame contenedor para centrar el contenido
    container = tk.Frame(root, bg="#f4f6fb")
    container.pack(expand=True, fill="both")

    main_frame = tk.Frame(container, bg="#ffffff", bd=2, relief="groove")
    main_frame.pack(expand=True)

    # Título grande
    titulo = tk.Label(main_frame, text="Predicción de Rendimiento Estudiantil", font=("Arial", 18, "bold"), bg="#ffffff", fg="#2d3e50")
    titulo.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # Definición de opciones para campos categóricos
    opciones = {
        "Parental_Involvement": ["Bajo", "Medio", "Alto"],
        "Access_to_Resources": ["Bajo", "Medio", "Alto"],
        "Extracurricular_Activities": ["Sí", "No"],
        "Motivation_Level": ["Bajo", "Medio", "Alto"],
        "Internet_Access": ["Sí", "No"],
        "Family_Income": ["Bajo", "Medio", "Alto"],
        "Teacher_Quality": ["Bajo", "Medio", "Alto"],
        "School_Type": ["Pública", "Privada"],
        "Peer_Influence": ["Positiva", "Neutral", "Negativa"],
        "Learning_Disabilities": ["Sí", "No"],
        "Parental_Education_Level": ["Secundaria", "Universidad", "Posgrado"],
        "Distance_from_Home": ["Cerca", "Moderada", "Lejos"],
        "Gender": ["Masculino", "Femenino"]
    }

    # Diccionario para mostrar los nombres de los campos en español
    campos_es = {
        "Parental_Involvement": "Participación parental",
        "Access_to_Resources": "Acceso a recursos",
        "Extracurricular_Activities": "Actividades extracurriculares",
        "Motivation_Level": "Nivel de motivación",
        "Internet_Access": "Acceso a internet",
        "Family_Income": "Ingresos familiares",
        "Teacher_Quality": "Calidad del docente",
        "School_Type": "Tipo de escuela",
        "Peer_Influence": "Influencia de compañeros",
        "Learning_Disabilities": "Discapacidades de aprendizaje",
        "Parental_Education_Level": "Nivel educativo de los padres",
        "Distance_from_Home": "Distancia desde casa",
        "Gender": "Género"
    }

    # Diccionario de descripciones en español para cada campo
    campos_desc = {
        "Hours_Studied": "Número de horas dedicadas al estudio por semana.",
        "Attendance": "Porcentaje de clases asistidas.",
        "Parental_Involvement": "Nivel de participación de los padres en la educación (Bajo, Medio, Alto).",
        "Access_to_Resources": "Disponibilidad de recursos educativos (Bajo, Medio, Alto).",
        "Extracurricular_Activities": "Participación en actividades extracurriculares (Sí, No).",
        "Sleep_Hours": "Promedio de horas de sueño por noche.",
        "Previous_Scores": "Puntajes obtenidos en exámenes previos.",
        "Motivation_Level": "Nivel de motivación del estudiante (Bajo, Medio, Alto).",
        "Internet_Access": "Disponibilidad de acceso a internet (Sí, No).",
        "Tutoring_Sessions": "Cantidad de sesiones de tutoría asistidas por mes.",
        "Family_Income": "Nivel de ingresos familiares (Bajo, Medio, Alto).",
        "Teacher_Quality": "Calidad de los docentes (Bajo, Medio, Alto).",
        "School_Type": "Tipo de escuela a la que asiste (Pública, Privada).",
        "Peer_Influence": "Influencia de los compañeros en el rendimiento (Positiva, Neutral, Negativa).",
        "Physical_Activity": "Promedio de horas de actividad física por semana.",
        "Learning_Disabilities": "Presencia de discapacidades de aprendizaje (Sí, No).",
        "Parental_Education_Level": "Nivel educativo más alto de los padres (Secundaria, Universidad, Posgrado).",
        "Distance_from_Home": "Distancia desde casa a la escuela (Cerca, Moderada, Lejos).",
        "Gender": "Género del estudiante (Masculino, Femenino)."
    }

    entradas = {}
    num_campos = len(campos)
    num_columnas = 2
    campos_por_col = (num_campos + 1) // num_columnas

    for idx, campo in enumerate(campos):
        col = idx // campos_por_col
        row = (idx % campos_por_col) * 2 + 1
        nombre_es = campos_es.get(campo, campo)
        tk.Label(main_frame, text=nombre_es, bg="#ffffff", fg="#2d3e50", font=("Arial", 10, "bold")).grid(row=row, column=col*2, sticky="w", pady=(3,0), padx=5)
        if campo in opciones:
            entradas[campo] = ttk.Combobox(main_frame, values=opciones[campo], state="readonly", font=("Arial", 10))
            entradas[campo].current(0)
        else:
            entradas[campo] = tk.Entry(main_frame, font=("Arial", 10))
        entradas[campo].grid(row=row, column=col*2+1, pady=(3,0), padx=5)
        # Descripción debajo del campo
        desc = campos_desc.get(campo, "")
        if desc:
            tk.Label(main_frame, text=desc, bg="#ffffff", fg="#888888", font=("Arial", 8, "italic")).grid(row=row+1, column=col*2, columnspan=2, sticky="w", padx=5, pady=(0,5))

    # Ajustar el resultado y el botón para que estén centrados debajo de ambas columnas
    resultado_var = tk.StringVar()
    resultado_label = tk.Label(main_frame, textvariable=resultado_var, font=("Arial", 14, "bold"), bg="#eaf6e7", fg="#27ae60", bd=2, relief="solid", padx=10, pady=5)
    resultado_label.grid(row=campos_por_col*2+2, column=0, columnspan=4, pady=15)

    # Botón estilizado
    style = ttk.Style()
    style.theme_use('default')
    style.configure("TButton",
        font=("Arial", 12, "bold"),
        foreground="#ffffff",
        background="#2980b9",
        borderwidth=1,
        focusthickness=3,
        focuscolor='none'
    )
    style.map("TButton",
        background=[('active', '#1c5d99'), ('!disabled', '#2980b9')],
        foreground=[('active', '#ffffff'), ('!disabled', '#ffffff')]
    )

    # Diccionario para mapear de español a inglés para cada campo categórico
    mapeo_es_en = {
        "Parental_Involvement": {"Bajo": "Low", "Medio": "Medium", "Alto": "High"},
        "Access_to_Resources": {"Bajo": "Low", "Medio": "Medium", "Alto": "High"},
        "Extracurricular_Activities": {"Sí": "Yes", "No": "No"},
        "Motivation_Level": {"Bajo": "Low", "Medio": "Medium", "Alto": "High"},
        "Internet_Access": {"Sí": "Yes", "No": "No"},
        "Family_Income": {"Bajo": "Low", "Medio": "Medium", "Alto": "High"},
        "Teacher_Quality": {"Bajo": "Low", "Medio": "Medium", "Alto": "High"},
        "School_Type": {"Pública": "Public", "Privada": "Private"},
        "Peer_Influence": {"Positiva": "Positive", "Neutral": "Neutral", "Negativa": "Negative"},
        "Learning_Disabilities": {"Sí": "Yes", "No": "No"},
        "Parental_Education_Level": {"Secundaria": "High School", "Universidad": "College", "Posgrado": "Postgraduate"},
        "Distance_from_Home": {"Cerca": "Near", "Moderada": "Moderate", "Lejos": "Far"},
        "Gender": {"Masculino": "Male", "Femenino": "Female"}
    }

    # Guardar los mapeos de codificación para cada campo categórico
    mapeos_codificacion = {}
    for col in opciones:
        encoder = LabelEncoder()
        encoder.fit([mapeo_es_en[col][v] if v in mapeo_es_en[col] else v for v in opciones[col]])
        mapeos_codificacion[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    def predecir():
        datos = []
        for campo in campos:
            if campo in opciones:
                valor = entradas[campo].get()
                # Convertir de español a inglés si aplica
                if campo in mapeo_es_en and valor in mapeo_es_en[campo]:
                    valor_en = mapeo_es_en[campo][valor]
                else:
                    valor_en = valor
                # Codificar usando el mapeo del encoder
                valor_cod = mapeos_codificacion[campo].get(valor_en)
                if valor_cod is None:
                    resultado_var.set("")
                    messagebox.showerror("Error", f"Valor no válido para {campos_es.get(campo, campo)}.")
                    return
                valor = valor_cod
            else:
                valor = entradas[campo].get()
                try:
                    valor = float(valor)
                except ValueError:
                    resultado_var.set("")
                    messagebox.showerror("Error", f"El campo '{campos_es.get(campo, campo)}' debe ser un número válido.")
                    return
            datos.append(valor)
        df_nuevo = pd.DataFrame([datos], columns=campos)
        X_nuevo = scaler.transform(df_nuevo)
        # Usar predict_proba para ajustar el umbral de 'Bajo'
        proba = rf.predict_proba(X_nuevo)[0]
        # Usar el mapeo real para encontrar el índice de la clase 'Bajo' según el LabelEncoder
        idx_bajo = list(rf.classes_).index(mapeo_rend['Bajo']) if mapeo_rend['Bajo'] in rf.classes_ else 0
        if proba[idx_bajo] > 0.3:
            resultado = "Bajo"
        else:
            pred = rf.predict(X_nuevo)[0]
            # Invertir el mapeo para obtener el nombre correcto de la clase
            mapeo_resultado = {v: k for k, v in mapeo_rend.items()}
            resultado = mapeo_resultado.get(pred, pred)
        resultado_var.set(f"Predicción: {resultado}")

    btn = ttk.Button(main_frame, text="Predecir", command=predecir, style="TButton")
    btn.grid(row=campos_por_col*2+3, column=0, columnspan=4, pady=10)

    root.mainloop()
