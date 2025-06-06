import pandas as pd
from app import rf, scaler, campos, opciones, mapeos_codificacion
from sklearn.preprocessing import LabelEncoder

# Datos de ejemplo para un caso "Bajo"
# Ajusta estos valores si tienes un caso real que debería ser "Bajo"
datos_bajo = {
    "Hours_Studied": 2,  # Dentro del rango sintético (1-5)
    "Attendance": 35,    # Dentro del rango sintético (30-60)
    "Parental_Involvement": mapeos_codificacion["Parental_Involvement"]["Low"],
    "Access_to_Resources": mapeos_codificacion["Access_to_Resources"]["Low"],
    "Extracurricular_Activities": mapeos_codificacion["Extracurricular_Activities"]["No"],
    "Sleep_Hours": 4,    # Dentro del rango sintético (3-5)
    "Previous_Scores": 30, # Dentro del rango sintético (20-51)
    "Motivation_Level": mapeos_codificacion["Motivation_Level"]["Low"],
    "Internet_Access": mapeos_codificacion["Internet_Access"]["No"],
    "Tutoring_Sessions": 0,
    "Family_Income": mapeos_codificacion["Family_Income"]["Low"],
    "Teacher_Quality": mapeos_codificacion["Teacher_Quality"]["Low"],
    "School_Type": mapeos_codificacion["School_Type"]["Public"],
    "Peer_Influence": mapeos_codificacion["Peer_Influence"]["Negative"],
    "Physical_Activity": 0,
    "Learning_Disabilities": mapeos_codificacion["Learning_Disabilities"]["Yes"],
    "Parental_Education_Level": mapeos_codificacion["Parental_Education_Level"]["High School"],
    "Distance_from_Home": mapeos_codificacion["Distance_from_Home"]["Far"],
    "Gender": mapeos_codificacion["Gender"]["Male"]
}

def test_prediccion_bajo():
    df_test = pd.DataFrame([datos_bajo], columns=campos)
    # Codificar variables categóricas igual que en el entrenamiento
    for col in df_test.select_dtypes(include="object").columns:
        if col in mapeos_codificacion:
            # Invertir el mapeo para decodificar si es necesario
            inv_map = {v: k for k, v in mapeos_codificacion[col].items()}
            df_test[col] = df_test[col].map(inv_map)
            df_test[col] = LabelEncoder().fit([k for k in mapeos_codificacion[col].keys()]).transform(df_test[col].astype(str))
    X_test = scaler.transform(df_test)
    proba = rf.predict_proba(X_test)[0]
    idx_bajo = list(rf.classes_).index(0) if 0 in rf.classes_ else 0
    if proba[idx_bajo] > 0.3:
        resultado = "Bajo"
    else:
        pred = rf.predict(X_test)[0]
        mapeo_resultado = {0: "Bajo", 1: "Medio", 2: "Alto"}
        resultado = mapeo_resultado.get(pred, pred)
    print(f"Predicción esperada: Bajo | Predicción obtenida: {resultado}")
    assert resultado == "Bajo", f"La predicción debería ser 'Bajo', pero fue '{resultado}'"

if __name__ == "__main__":
    test_prediccion_bajo()
    print("Test completado correctamente.")
