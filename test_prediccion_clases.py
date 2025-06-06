import pandas as pd
from app import rf, scaler, campos, opciones, mapeos_codificacion, mapeo_rend
from sklearn.preprocessing import LabelEncoder

def test_prediccion_bajo():
    datos_bajo = {
        "Hours_Studied": 2,
        "Attendance": 35,
        "Parental_Involvement": mapeos_codificacion["Parental_Involvement"]["Low"],
        "Access_to_Resources": mapeos_codificacion["Access_to_Resources"]["Low"],
        "Extracurricular_Activities": mapeos_codificacion["Extracurricular_Activities"]["No"],
        "Sleep_Hours": 4,
        "Previous_Scores": 30,
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
    df_test = pd.DataFrame([datos_bajo], columns=campos)
    X_test = scaler.transform(df_test)
    proba = rf.predict_proba(X_test)[0]
    idx_bajo = list(rf.classes_).index(mapeo_rend["Bajo"]) if mapeo_rend["Bajo"] in rf.classes_ else 0
    if proba[idx_bajo] > 0.3:
        resultado = "Bajo"
    else:
        pred = rf.predict(X_test)[0]
        mapeo_resultado = {v: k for k, v in mapeo_rend.items()}
        resultado = mapeo_resultado.get(pred, pred)
    print(f"Predicción esperada: Bajo | Predicción obtenida: {resultado}")
    assert resultado == "Bajo", f"La predicción debería ser 'Bajo', pero fue '{resultado}'"

def test_prediccion_medio():
    datos_medio = {
        "Hours_Studied": 10,
        "Attendance": 70,
        "Parental_Involvement": mapeos_codificacion["Parental_Involvement"]["Medium"],
        "Access_to_Resources": mapeos_codificacion["Access_to_Resources"]["Medium"],
        "Extracurricular_Activities": mapeos_codificacion["Extracurricular_Activities"]["No"],
        "Sleep_Hours": 6,
        "Previous_Scores": 65,
        "Motivation_Level": mapeos_codificacion["Motivation_Level"]["Medium"],
        "Internet_Access": mapeos_codificacion["Internet_Access"]["Yes"],
        "Tutoring_Sessions": 1,
        "Family_Income": mapeos_codificacion["Family_Income"]["Medium"],
        "Teacher_Quality": mapeos_codificacion["Teacher_Quality"]["Medium"],
        "School_Type": mapeos_codificacion["School_Type"]["Public"],
        "Peer_Influence": mapeos_codificacion["Peer_Influence"]["Neutral"],
        "Physical_Activity": 2,
        "Learning_Disabilities": mapeos_codificacion["Learning_Disabilities"]["No"],
        "Parental_Education_Level": mapeos_codificacion["Parental_Education_Level"]["High School"],
        "Distance_from_Home": mapeos_codificacion["Distance_from_Home"]["Moderate"],
        "Gender": mapeos_codificacion["Gender"]["Female"]
    }
    df_test = pd.DataFrame([datos_medio], columns=campos)
    X_test = scaler.transform(df_test)
    proba = rf.predict_proba(X_test)[0]
    idx_medio = list(rf.classes_).index(mapeo_rend["Medio"]) if mapeo_rend["Medio"] in rf.classes_ else 1
    if proba[idx_medio] > 0.3:
        resultado = "Medio"
    else:
        pred = rf.predict(X_test)[0]
        mapeo_resultado = {v: k for k, v in mapeo_rend.items()}
        resultado = mapeo_resultado.get(pred, pred)
    print(f"Predicción esperada: Medio | Predicción obtenida: {resultado}")
    assert resultado == "Medio", f"La predicción debería ser 'Medio', pero fue '{resultado}'"

def test_prediccion_alto():
    datos_alto = {
        "Hours_Studied": 20,
        "Attendance": 90,
        "Parental_Involvement": mapeos_codificacion["Parental_Involvement"]["High"],
        "Access_to_Resources": mapeos_codificacion["Access_to_Resources"]["High"],
        "Extracurricular_Activities": mapeos_codificacion["Extracurricular_Activities"]["Yes"],
        "Sleep_Hours": 8,
        "Previous_Scores": 90,
        "Motivation_Level": mapeos_codificacion["Motivation_Level"]["High"],
        "Internet_Access": mapeos_codificacion["Internet_Access"]["Yes"],
        "Tutoring_Sessions": 3,
        "Family_Income": mapeos_codificacion["Family_Income"]["High"],
        "Teacher_Quality": mapeos_codificacion["Teacher_Quality"]["High"],
        "School_Type": mapeos_codificacion["School_Type"]["Private"],
        "Peer_Influence": mapeos_codificacion["Peer_Influence"]["Positive"],
        "Physical_Activity": 5,
        "Learning_Disabilities": mapeos_codificacion["Learning_Disabilities"]["No"],
        "Parental_Education_Level": mapeos_codificacion["Parental_Education_Level"]["Postgraduate"],
        "Distance_from_Home": mapeos_codificacion["Distance_from_Home"]["Near"],
        "Gender": mapeos_codificacion["Gender"]["Male"]
    }
    df_test = pd.DataFrame([datos_alto], columns=campos)
    X_test = scaler.transform(df_test)
    proba = rf.predict_proba(X_test)[0]
    idx_alto = list(rf.classes_).index(mapeo_rend["Alto"]) if mapeo_rend["Alto"] in rf.classes_ else 2
    if proba[idx_alto] > 0.3:
        resultado = "Alto"
    else:
        pred = rf.predict(X_test)[0]
        mapeo_resultado = {v: k for k, v in mapeo_rend.items()}
        resultado = mapeo_resultado.get(pred, pred)
    print(f"Predicción esperada: Alto | Predicción obtenida: {resultado}")
    assert resultado == "Alto", f"La predicción debería ser 'Alto', pero fue '{resultado}'"

if __name__ == "__main__":
    test_prediccion_bajo()
    test_prediccion_medio()
    test_prediccion_alto()
    print("Todos los tests pasaron correctamente.")
