import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from model import NeuralNetwork
import json
import os

# Configuración de FastAPI
app = FastAPI(title="THE TICKER API")

# CORS: permite que Angular (puerto 4200) hable con Python (puerto 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de los componentes de IA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cargando SentenceTransformer...")
vectorizer = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

print("Cargando Red Neuronal...")
model = NeuralNetwork().to(device)

model_path = "intent_classifier.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Modelo cargado desde: {model_path}")
else:
    print("WARNING: No se encontró 'intent_classifier.pth'. Ejecuta train.py primero.")

# Mapa de etiquetas
labels_map = {}
if os.path.exists("labels.json"):
    with open("labels.json", "r") as f:
        labels_map = json.load(f)

# Esquemas de datos
class MessageRequest(BaseModel):
    message: str

# Endpoints
@app.get("/")
def health():
    return {"status": "online", "device": str(device)}

@app.post("/chat")
async def chat(request: MessageRequest):
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    try:
        # 1. Vectorizar texto → 512 dimensiones
        embedding = vectorizer.encode([text])
        tensor_x = torch.tensor(embedding, dtype=torch.float32).to(device)

        # 2. Predicción
        with torch.no_grad():
            outputs = model(tensor_x)
            _, predicted = torch.max(outputs, 1)
            intent_id = str(predicted.item())

        # 3. Traducir ID → nombre legible
        intent_name = labels_map.get(intent_id, f"intent_{intent_id}")

        return {
            "intent": intent_name,
            "response": f"Intención detectada: {intent_name}",
            "status": "success"
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# Arranque
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
