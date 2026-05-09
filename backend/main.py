from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import ChatModel

app = FastAPI(title="Chatbot IA en Español (OpenAssistant)")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

# Variable global para el modelo
chat_model = None

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
SAVED_MODEL_DIR = REPO_DIR / "saved_chat_model"
CUSTOM_MODEL_PATH = SAVED_MODEL_DIR / "custom_model.pth"

@app.on_event("startup")
async def load_model():
    global chat_model
    try:
        tokenizer_path = str(SAVED_MODEL_DIR) if (SAVED_MODEL_DIR / "tokenizer.json").exists() else "datificate/gpt2-small-spanish"

        if CUSTOM_MODEL_PATH.exists():
            print("Cargando red neuronal propia entrenada localmente...")
            chat_model = ChatModel(vocab_model_name=tokenizer_path, load_path=str(CUSTOM_MODEL_PATH))
        else:
            print("Instanciando red neuronal propia desde cero (sin entrenar)...")
            print(f"No se ha encontrado el fichero de pesos: {CUSTOM_MODEL_PATH}")
            chat_model = ChatModel(vocab_model_name=tokenizer_path)
            
        print("Modelo de lenguaje listo para conversar.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")

@app.get("/")
async def root():
    return {"message": "API de Chatbot funcionando. Preparada para usar un modelo entrenado con OpenAssistant oasst1 en español."}

@app.get("/health")
async def health():
    return {
        "status": "ok" if chat_model is not None else "model_not_loaded",
        "custom_model_found": CUSTOM_MODEL_PATH.exists(),
    }

@app.post("/chat")
async def chat(req: MessageRequest):
    if chat_model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
        
    try:
        response = chat_model.generate_response(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
