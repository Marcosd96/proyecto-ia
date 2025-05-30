# Predicción de Rendimiento Estudiantil

Este proyecto es una aplicación de escritorio en Python que predice el rendimiento estudiantil (Bajo, Medio, Alto) a partir de factores socioeconómicos y académicos, usando un modelo de Random Forest y una interfaz gráfica con Tkinter.

## Archivos principales
- `app.py`: Código principal de la aplicación y la interfaz gráfica.
- `StudentPerformanceFactors.csv`: Dataset necesario para entrenar y usar el modelo.

## Requisitos
- Python 3.8+
- Paquetes: pandas, scikit-learn, imbalanced-learn, tkinter

Puedes instalar las dependencias con:

```
pip install -r requirements.txt
```

## Uso

1. Asegúrate de tener el archivo `StudentPerformanceFactors.csv` en la misma carpeta que `app.py`.
2. Ejecuta la aplicación:

```
python app.py
```

3. Ingresa los datos solicitados en la interfaz y presiona "Predecir" para ver el resultado.

## Notas
- No subas archivos de la carpeta `build/` ni archivos `.pyc` al repositorio.
- Si necesitas compilar a ejecutable, puedes usar PyInstaller.
