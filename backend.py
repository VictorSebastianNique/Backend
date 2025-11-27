from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import requests
import zipfile

# --- CONFIGURACIÓN ---
MODEL_URL = "https://github.com/VictorSebastianNique/Backend/releases/download/mi_model/emotion_model_final.zip"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/emotion_model_final.keras"
CLASSES_PATH = f"{MODEL_DIR}/classes.txt"

app = FastAPI(title="Sistema Neuro-Psicológico")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []


# --- FUNCIÓN: Descargar ZIP desde GitHub ---
def download_model_zip():
    zip_path = f"{MODEL_DIR}/model.zip"

    print("Descargando modelo desde GitHub Releases...")
    r = requests.get(MODEL_URL, stream=True)

    if r.status_code != 200:
        raise Exception("Error al descargar el modelo desde GitHub")

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Modelo descargado correctamente.")

    print("Descomprimiendo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(zip_path)
    print("Modelo descomprimido y ZIP eliminado.")


# --- EVENTO: Cargar o descargar recursos al iniciar ---
@app.on_event("startup")
def load_resources():
    global model, class_names

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Si no existe el modelo, lo descarga automáticamente
    if not os.path.exists(MODEL_PATH):
        download_model_zip()

    # Carga del modelo
    print("Cargando modelo .keras...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Cargar clases
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            class_names = f.read().splitlines()
    else:
        class_names = ["Clase_0", "Clase_1"]  # Evita crashear si no hay archivo

    print("Modelo y clases cargados correctamente.")


# --- ENDPOINT DE PREDICCIÓN ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor.")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de imagen inválido")

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    result_idx = np.argmax(predictions[0])
    label = class_names[result_idx]
    confidence = float(np.max(predictions[0]))

    full_stats = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

    return {
        "emocion_detectada": label,
        "confianza": confidence,
        "analisis_detallado": full_stats
    }
