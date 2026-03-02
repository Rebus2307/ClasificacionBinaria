# ==============================
# EJERCICIO CLASIFICACIÓN BINARIA - DIABETES
# ==============================

# Importamos librerías necesarias
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==============================
# 1️⃣ CARGAR DATASET
# ==============================

# Cargar la base de datos
data = pd.read_csv("diabetes.csv")

# Seleccionamos solo las columnas que usaremos
X = data[["BMI", "Age", "PlasmaGlucose"]]  # Variables predictoras
y = data["Diabetic"]  # Variable objetivo (0 o 1)

# ==============================
# 2️⃣ PREPROCESAMIENTO
# ==============================

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalamos los datos (MUY importante en regresión logística)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 3️⃣ ENTRENAMIENTO DEL MODELO
# ==============================

# Creamos el modelo de Regresión Logística
model = LogisticRegression()

# Entrenamos el modelo
model.fit(X_train, y_train)

# ==============================
# 4️⃣ INTERFAZ GRÁFICA
# ==============================

# Función que se ejecuta cuando el usuario presiona el botón
def predecir():
    try:
        # Obtener valores ingresados por el usuario
        bmi = float(entry_bmi.get())
        age = float(entry_age.get())
        glucose = float(entry_glucose.get())

        # Crear arreglo con los datos
        paciente = np.array([[bmi, age, glucose]])

        # Escalar los datos igual que el entrenamiento
        paciente = scaler.transform(paciente)

        # Obtener probabilidad
        probabilidad = model.predict_proba(paciente)[0][1]

        # Obtener clasificación final (0 o 1)
        prediccion = model.predict(paciente)[0]

        # Mostrar resultados
        resultado_label.config(
            text=f"Probabilidad de Diabetes: {probabilidad:.2f}\nClasificación Final: {prediccion}"
        )

    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")


# Crear ventana principal
ventana = tk.Tk()
ventana.title("Predicción de Diabetes")
ventana.geometry("400x350")

# Título
titulo = tk.Label(ventana, text="Clasificación Binaria - Diabetes", font=("Arial", 14))
titulo.pack(pady=10)

# ==============================
# CAMPOS DE ENTRADA
# ==============================

# BMI
tk.Label(ventana, text="BMI:").pack()
entry_bmi = tk.Entry(ventana)
entry_bmi.pack()

# Age
tk.Label(ventana, text="Age:").pack()
entry_age = tk.Entry(ventana)
entry_age.pack()

# Plasma Glucose
tk.Label(ventana, text="Plasma Glucose:").pack()
entry_glucose = tk.Entry(ventana)
entry_glucose.pack()

# Botón de predicción
boton = tk.Button(ventana, text="Predecir", command=predecir)
boton.pack(pady=15)

# Label para mostrar resultado
resultado_label = tk.Label(ventana, text="", font=("Arial", 12))
resultado_label.pack()

# Ejecutar la interfaz
ventana.mainloop()