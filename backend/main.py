from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from model import ChatModel

app = FastAPI(title="THE TICKER API")

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

@app.on_event("startup")
async def load_model():
    global chat_model
    try:
        model_path = "./saved_chat_model/custom_model.pth"
        if os.path.exists(model_path):
            print("Cargando modelo ajustado guardado localmente...")
            chat_model = ChatModel(vocab_model_name="./saved_chat_model", load_path=model_path)
        else:
            print("Cargando modelo preentrenado en espanol...")
            chat_model = ChatModel(vocab_model_name="datificate/gpt2-small-spanish")
            
        print("Modelo de lenguaje listo para conversar.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")

@app.get("/")
async def root():
    return {"message": "API de THE TICKER funcionando. Carga CausalTransformer entrenado si existe; si no, usa un modelo preentrenado en espanol."}

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
