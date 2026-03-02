# ==========================================================
# SISTEMA DE CLASIFICACIÓN BINARIA - PREDICCIÓN DE DIABETES
# Incluye:
# - Regresión Logística
# - Interfaz gráfica en español
# - Probabilidad
# - Diagnóstico final
# - Gráfica dinámica del paciente
# ==========================================================

# =============================
# IMPORTAR LIBRERÍAS
# =============================
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================
# CARGAR DATASET
# =============================
data = pd.read_csv("diabetes.csv")

# Variables predictoras
X = data[["BMI", "Age", "PlasmaGlucose"]]

# Variable objetivo (0 = No Diabético, 1 = Diabético)
y = data["Diabetic"]

# =============================
# PREPROCESAMIENTO
# =============================

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# ENTRENAR MODELO
# =============================
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# =============================
# FUNCIÓN SIGMOIDE
# =============================
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# =============================
# FUNCIÓN DE PREDICCIÓN
# =============================
def predecir():
    try:
        # Obtener valores del usuario
        imc = float(entrada_imc.get())
        edad = float(entrada_edad.get())
        glucosa = float(entrada_glucosa.get())

        # Crear arreglo del paciente
        paciente = np.array([[imc, edad, glucosa]])

        # Escalar igual que entrenamiento
        paciente_escalado = scaler.transform(paciente)

        # Obtener probabilidad de clase 1 (Diabetes)
        probabilidad = modelo.predict_proba(paciente_escalado)[0][1]

        # Obtener clasificación final
        prediccion = modelo.predict(paciente_escalado)[0]

        # Interpretar resultado
        diagnostico = "Diabético" if prediccion == 1 else "No Diabético"

        # Mostrar resultado en interfaz
        resultado_label.config(
            text=f"Probabilidad de Diabetes: {probabilidad:.2f}\n"
                 f"Diagnóstico Final: {diagnostico}"
        )

        # =============================
        # PARTE GRÁFICA
        # =============================

        # Obtener valor lineal (z) antes de sigmoide
        z_paciente = modelo.decision_function(paciente_escalado)[0]

        # Generar curva sigmoide
        z = np.linspace(-10, 10, 400)
        p = sigmoide(z)

        # Crear gráfica
        plt.figure()
        plt.plot(z, p)
        plt.scatter(z_paciente, probabilidad)

        # Línea del umbral 0.5
        plt.axhline(y=0.5)

        plt.title("Curva Sigmoide - Clasificación del Paciente")
        plt.xlabel("Valor lineal (z)")
        plt.ylabel("Probabilidad")
        plt.show()

    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")

# =============================
# CREAR INTERFAZ GRÁFICA
# =============================
ventana = tk.Tk()
ventana.title("Sistema de Predicción de Diabetes")
ventana.geometry("420x400")
ventana.resizable(False, False)

# Título
titulo = tk.Label(
    ventana,
    text="Sistema de Clasificación Binaria\nPredicción de Diabetes",
    font=("Arial", 14)
)
titulo.pack(pady=15)

# =============================
# CAMPOS DE ENTRADA
# =============================

tk.Label(ventana, text="IMC (BMI):").pack()
entrada_imc = tk.Entry(ventana)
entrada_imc.pack()

tk.Label(ventana, text="Edad:").pack()
entrada_edad = tk.Entry(ventana)
entrada_edad.pack()

tk.Label(ventana, text="Glucosa en Plasma:").pack()
entrada_glucosa = tk.Entry(ventana)
entrada_glucosa.pack()

# Botón
boton = tk.Button(
    ventana,
    text="Calcular Diagnóstico",
    command=predecir
)
boton.pack(pady=20)

# Resultado
resultado_label = tk.Label(
    ventana,
    text="",
    font=("Arial", 12)
)
resultado_label.pack()

# Ejecutar ventana
ventana.mainloop()