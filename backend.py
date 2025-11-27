from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf  # Usaremos el intérprete Lite
import numpy as np
from PIL import Image
import io
import os

# --- CONFIGURACIÓN ---
# Ahora usamos el modelo ligero
MODEL_PATH = "models/emotion_model.tflite"
CLASSES_PATH = "models/classes.txt"

app = FastAPI(title="Sistema Neuro-Psicológico Lite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
interpreter = None
input_details = None
output_details = None
class_names = []

@app.on_event("startup")
def load_resources():
    global interpreter, input_details, output_details, class_names
    
    # 1. Cargar Clases
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            class_names = f.read().splitlines()
    else:
        print("ERROR: No se encontró classes.txt")

    # 2. Cargar Modelo TFLite (Bajo consumo de RAM)
    if os.path.exists(MODEL_PATH):
        print(f"Cargando TFLite desde {MODEL_PATH}...")
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors() # Reserva solo la memoria necesaria
            
            # Obtener referencias de entrada/salida
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("Modelo Lite cargado exitosamente.")
        except Exception as e:
            print(f"Error fatal cargando modelo: {e}")
    else:
        print(f"ADVERTENCIA: No se encontró el modelo en {MODEL_PATH}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    # 1. Procesar Imagen
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    # 2. Preprocesamiento Matemático
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0) # Batch de 1

    # 3. Inferencia con TFLite (Manual)
    # Poner datos en la "neurona" de entrada
    interpreter.set_tensor(input_details[0]['index'], img_array)
    # Ejecutar la red
    interpreter.invoke()
    # Leer datos de la "neurona" de salida
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # 4. Resultado
    result_idx = np.argmax(predictions)
    label = class_names[result_idx]
    confidence = float(predictions[result_idx])
    
    full_stats = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    return {
        "emocion_detectada": label,
        "confianza": confidence,
        "analisis_detallado": full_stats
    }
