from pathlib import Path
import unicodedata

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

TOPICS = [
    (
        ("interes compuesto",),
        ("interes compuesto", "intereses sobre intereses"),
        "El interes compuesto es ganar intereses sobre intereses. Por ejemplo, si ahorras 100 euros y generas un 5 %, pasas a 105; "
        "despues el calculo se hace sobre 105. Es una explicacion educativa, no una promesa de beneficio.",
    ),
    (
        ("presupuesto", "50/30/20", "gastos"),
        ("presupuesto", "gastos", "50/30/20"),
        "Para organizar un presupuesto mensual, anota ingresos y gastos reales. Separa gastos fijos, gastos variables y ahorro. "
        "La regla 50/30/20 puede servir como guia inicial para revisar si el dinero se va sin control.",
    ),
    (
        ("etf",),
        ("etf", "cotiza", "diversificacion"),
        "Un ETF es un fondo que cotiza en bolsa y agrupa varios activos. Sirve como ejemplo de diversificacion, porque no dependes "
        "de una sola empresa. Aun asi, invertir implica riesgo y no debo recomendar productos concretos.",
    ),
    (
        ("ahorrar", "ahorro", "estudiante"),
        ("ahorro", "fondo de emergencia", "habito"),
        "Empieza con una cantidad pequena y constante. Separa el ahorro al recibir dinero y crea primero un fondo de emergencia. "
        "Lo importante al principio es el habito, no una cantidad perfecta.",
    ),
    (
        ("estafa", "phishing", "seguridad"),
        ("estafa", "desconfia", "claves", "banco"),
        "Desconfia de promesas de dinero rapido, presion para decidir ya y enlaces inesperados. No compartas claves ni codigos. "
        "Si dudas, contacta con tu banco por canales oficiales.",
    ),
    (
        ("deuda", "prestamo", "credito"),
        ("bola de nieve", "mayor coste", "cuota"),
        "Primero lista cada deuda con importe, cuota, plazo y tipo de interes. Despues prioriza las de mayor coste o usa el metodo "
        "bola de nieve, empezando por la mas pequena. Es una orientacion educativa, no asesoramiento profesional.",
    ),
]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def extract_user_question(message: str) -> str:
    marker = "Pregunta del usuario:"
    if marker in message:
        return message.split(marker, 1)[1]
    return message


def educational_response(message: str):
    question = normalize_text(extract_user_question(message))
    for question_terms, _, answer in TOPICS:
        if any(term in question for term in question_terms):
            return answer

    return None


def response_is_usable(response: str) -> bool:
    text = normalize_text(response)
    words = [word for word in text.split() if len(word) > 2]
    repeated_words = len(words) - len(set(words))

    return len(text) >= 90 and repeated_words <= 10 and not text.endswith((" de", " un", " una", " y", " o"))


def response_matches_question(message: str, response: str) -> bool:
    question = normalize_text(extract_user_question(message))
    answer = normalize_text(response)

    for question_terms, answer_terms, _ in TOPICS:
        if any(term in question for term in question_terms):
            return any(term in answer for term in answer_terms)

    return True


def clean_model_response(response: str) -> str:
    cleaned = response.replace("</s>", "").replace("</s", "").replace("</", "").replace("<pad>", "").strip()
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned


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
        response = clean_model_response(chat_model.generate_response(req.message))
        fallback = educational_response(req.message)
        if fallback and (not response_is_usable(response) or not response_matches_question(req.message, response)):
            response = fallback
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
