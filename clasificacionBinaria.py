#
# Sistema de Clasificación Binaria - Predicción de Diabetes
#
# Este script carga un dataset de diabetes, entrena un modelo de regresión
# logística y provee una interfaz gráfica sencilla (Tkinter) para ingresar
# los datos de un paciente y obtener la probabilidad de ser diabético.
#
# Comentarios: todo el código está anotado en español para explicar cada
# parte importante: carga de datos, preprocesamiento, entrenamiento,
# funciones de predicción y la interfaz gráfica.
#

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
# Cargar dataset desde el archivo CSV en el mismo directorio
data = pd.read_csv("diabetes.csv")

# Seleccionar las columnas que usaremos como predictores (features):
# - BMI: índice de masa corporal
# - Age: edad del paciente
# - PlasmaGlucose: nivel de glucosa en plasma
X = data[["BMI", "Age", "PlasmaGlucose"]]

# Variable objetivo (target): 0 = No Diabético, 1 = Diabético
y = data["Diabetic"]


# =============================
# PREPROCESAMIENTO
# =============================

# Separar los datos en conjuntos de entrenamiento y prueba.
# `test_size=0.2` reserva el 20% para evaluación y `random_state` fija
# la división para reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado de características: muchas técnicas de ML funcionan mejor si
# las variables están normalizadas/estandarizadas (media 0, desviación 1).
# Usamos `StandardScaler` y ajustamos (fit) sólo en los datos de entrenamiento
# para evitar filtrado de información del conjunto de prueba.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =============================
# ENTRENAR MODELO
# =============================

# Creamos una instancia de regresión logística y la entrenamos con los
# datos escalados. La regresión logística es adecuada para clasificación
# binaria y nos permite obtener probabilidades mediante `predict_proba`.
modelo = LogisticRegression()
modelo.fit(X_train, y_train)


# =============================
# FUNCIÓN SIGMOIDE
# =============================

def sigmoide(z):
    """Devuelve la función sigmoide para una entrada escalar o vector.

    La sigmoide transforma valores reales en el rango (0, 1), y es la
    función que interpreta el valor lineal del modelo como probabilidad.
    """
    return 1 / (1 + np.exp(-z))


# =============================
# FUNCIÓN DE PREDICCIÓN
# =============================

def predecir():
    """Recoge valores desde la interfaz, calcula probabilidad y muestra
    resultado. También genera una gráfica de la curva sigmoide y la
    posición del paciente sobre dicha curva.
    """
    try:
        # 1) Leer y convertir entradas de la interfaz. `get()` devuelve
        # cadenas, por eso convertimos a `float`. Si falla, cae al except.
        imc = float(entrada_imc.get())
        edad = float(entrada_edad.get())
        glucosa = float(entrada_glucosa.get())

        # 2) Construir la matriz de características del paciente con la
        # misma forma que el conjunto de entrenamiento (1 muestra, n features).
        paciente = np.array([[imc, edad, glucosa]])

        # 3) Escalar las características usando el `scaler` ajustado antes.
        # Es importante usar el mismo escalador que en el entrenamiento.
        paciente_escalado = scaler.transform(paciente)

        # 4) `predict_proba` devuelve probabilidades para cada clase;
        # el índice [0][1] es la probabilidad de la clase positiva (diabético).
        probabilidad = modelo.predict_proba(paciente_escalado)[0][1]

        # 5) `predict` devuelve la etiqueta (0 o 1) según el umbral por
        # defecto (0.5) aplicado internamente al modelo.
        prediccion = modelo.predict(paciente_escalado)[0]

        # 6) Interpretar la etiqueta en un texto legible.
        diagnostico = "Diabético" if prediccion == 1 else "No Diabético"

        # 7) Mostrar la probabilidad y el diagnóstico en la etiqueta del GUI.
        resultado_label.config(
            text=f"Probabilidad de Diabetes: {probabilidad:.2f}\n"
                 f"Diagnóstico Final: {diagnostico}"
        )

        # =============================
        # PARTE GRÁFICA (ilustrativa)
        # =============================

        # Obtiene el valor lineal (z) del modelo antes de aplicar sigmoide.
        # `decision_function` devuelve ese valor; aplicando la sigmoide
        # obtendremos la probabilidad (equivalente a predict_proba).
        z_paciente = modelo.decision_function(paciente_escalado)[0]

        # Generar una gama de valores z para dibujar la curva sigmoide.
        z = np.linspace(-10, 10, 400)
        p = sigmoide(z)

        # Dibujar la curva y marcar la posición del paciente.
        plt.figure()
        plt.plot(z, p, label="Sigmoide")
        plt.scatter(z_paciente, probabilidad, color="red", label="Paciente")

        # Línea horizontal en 0.5 para visualizar el umbral típico.
        plt.axhline(y=0.5, color="gray", linestyle="--", label="Umbral 0.5")

        plt.title("Curva Sigmoide - Clasificación del Paciente")
        plt.xlabel("Valor lineal (z)")
        plt.ylabel("Probabilidad")
        plt.legend()
        plt.show()

    except ValueError:
        # Si el usuario introduce algo que no puede convertirse a float,
        # se muestra un mensaje de error amigable.
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")


# =============================
# CREAR INTERFAZ GRÁFICA (Tkinter)
# =============================

# Inicializar la ventana principal de la aplicación.
ventana = tk.Tk()
ventana.title("Sistema de Predicción de Diabetes")
ventana.geometry("420x400")
ventana.resizable(False, False)

# Etiqueta de título con texto descriptivo.
titulo = tk.Label(
    ventana,
    text="Sistema de Clasificación Binaria\nPredicción de Diabetes",
    font=("Arial", 14)
)
titulo.pack(pady=15)

# Campo para IMC (BMI)
tk.Label(ventana, text="IMC (BMI):").pack()
entrada_imc = tk.Entry(ventana)
entrada_imc.pack()

# Campo para Edad
tk.Label(ventana, text="Edad:").pack()
entrada_edad = tk.Entry(ventana)
entrada_edad.pack()

# Campo para Glucosa en plasma
tk.Label(ventana, text="Glucosa en Plasma:").pack()
entrada_glucosa = tk.Entry(ventana)
entrada_glucosa.pack()

# Botón que dispara la función `predecir` cuando el usuario lo presiona.
boton = tk.Button(
    ventana,
    text="Calcular Diagnóstico",
    command=predecir
)
boton.pack(pady=20)

# Etiqueta donde se mostrará el resultado (probabilidad y diagnóstico).
resultado_label = tk.Label(
    ventana,
    text="",
    font=("Arial", 12)
)
resultado_label.pack()

# Iniciar el loop principal de la interfaz gráfica.
ventana.mainloop()