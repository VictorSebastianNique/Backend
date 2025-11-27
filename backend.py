from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import requests  # Librería para descargar el modelo

# --- CONFIGURACIÓN ---
# PEGA AQUÍ EL LINK QUE COPIASTE DE GITHUB RELEASES:
MODEL_URL = "https://github.com/VictorSebastianNique/Backend/releases/download/mi_model_lite/emotion_model.zip"

# Rutas locales
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

def download_model():
    """Descarga el modelo desde GitHub Releases si no existe localmente"""
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Modelo no encontrado en {MODEL_PATH}. Descargando desde GitHub Releases...")
        
        # Crear carpeta models si no existe
        os.makedirs("models", exist_ok=True)
        
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() # Verificar si el link funciona
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Descarga completada exitosamente.")
        except Exception as e:
            print(f"❌ Error fatal descargando el modelo: {e}")
            raise e
    else:
        print("✅ El modelo ya existe localmente.")

@app.on_event("startup")
def load_resources():
    global interpreter, input_details, output_details, class_names
    
    # 1. Intentar descargar el modelo primero
    try:
        download_model()
    except Exception as e:
        print("El servidor iniciará sin IA debido a error de descarga.")
        return

    # 2. Cargar Clases (Crear archivo dummy si no existe para evitar crash)
    if not os.path.exists(CLASSES_PATH):
        # Fallback de emergencia
        print("Creando lista de clases por defecto...")
        os.makedirs("models", exist_ok=True)
        default_classes = ["alegria", "decepcion", "depresion", "enojo", "frustracion", "tristeza"]
        with open(CLASSES_PATH, "w") as f:
            f.write("\n".join(default_classes))
            
    with open(CLASSES_PATH, "r") as f:
        class_names = f.read().splitlines()

    # 3. Cargar Modelo TFLite en Memoria
    try:
        print(f"Cargando TFLite desde {MODEL_PATH}...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("🚀 Sistema de IA Operativo.")
    except Exception as e:
        print(f"Error cargando el intérprete TFLite: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="El modelo de IA aún no está listo o falló al cargar.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    # Preprocesamiento
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Resultado
    result_idx = np.argmax(predictions)
    label = class_names[result_idx]
    confidence = float(predictions[result_idx])
    
    full_stats = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    return {
        "emocion_detectada": label,
        "confianza": confidence,
        "analisis_detallado": full_stats
    }
