# THE TICKER - TFG DAM

THE TICKER es un asistente educativo de finanzas personales para jovenes. El usuario completa un onboarding sencillo, recibe una hoja de ruta inicial segun su perfil y despues conversa con un chatbot conectado a un backend de IA.

El objetivo del proyecto no es crear una plataforma financiera profesional, sino demostrar de forma clara conocimientos de frontend, backend, APIs, integracion con IA, gestion de estado, diseño de interfaz y despliegue local de un proyecto full-stack.

## Idea del Proyecto

La aplicacion responde a una necesidad concreta: muchas personas jovenes tienen dudas basicas sobre ahorro, deudas, presupuesto o inversion, pero no saben por donde empezar. THE TICKER convierte esas dudas en una experiencia guiada:

1. El frontend pregunta datos minimos del usuario.
2. La aplicacion genera un perfil educativo simple.
3. Se muestra una hoja de ruta con conceptos iniciales.
4. El usuario puede preguntar al chatbot en lenguaje natural.
5. El backend procesa la pregunta con un modelo de lenguaje en Python.

La IA se usa como apoyo conversacional, no como sustituto del aprendizaje tecnico. Por eso el proyecto mantiene una arquitectura sencilla y explicable.

## Reparto Tecnico

- **Frontend:** Angular, componentes standalone, signals, HTML/CSS responsive, onboarding, hoja de ruta, chat e integracion HTTP.
- **Backend:** FastAPI, Pydantic, CORS, endpoint `/chat`, carga del modelo y generacion de respuestas con PyTorch/Transformers.
- **IA:** `CausalTransformer` propio para explicar entrenamiento, dataset conversacional en espanol, tokenizer, prompt contextual y fallback con modelo preentrenado.

## Arquitectura

```text
Usuario
  -> Frontend Angular
      -> Onboarding + perfil + chat
      -> POST http://localhost:8000/chat
          -> Backend FastAPI
              -> ChatModel
                  -> CausalTransformer entrenado si existe
                  -> Modelo preentrenado si no hay pesos locales
```

## Requisitos Previos

- [Node.js](https://nodejs.org/) y **pnpm**
- [Python 3.10 - 3.12](https://www.python.org/) y **uv**
- Git

Si no tienes pnpm:

```bash
npm install -g pnpm
```

## Puesta en Marcha

### Frontend

```bash
cd frontend
pnpm install
pnpm start
```

Angular levantara la aplicacion normalmente en:

```text
http://localhost:4200
```

### Backend

```bash
cd backend
uv venv
uv sync
uv run python main.py
```

FastAPI levantara la API en:

```text
http://localhost:8000
```

### Entrenamiento Opcional

El entrenamiento no es necesario para levantar la demo, pero forma parte de la explicacion tecnica del backend:

```bash
cd backend
uv run python train.py
```

Este comando prepara conversaciones en espanol, entrena el `CausalTransformer` y guarda los pesos en `saved_chat_model/custom_model.pth`.

## Entorno Virtual del Backend

Activar en bash o zsh:

```bash
source .venv/bin/activate
```

Activar en fish:

```bash
source .venv/bin/activate.fish
```

Desactivar:

```bash
deactivate
```

## Limitaciones Reconocidas

- Las respuestas del chatbot deben entenderse como contenido educativo, no como asesoramiento financiero profesional.
- El perfil del usuario es intencionadamente simple para que la logica sea facil de explicar.
- El modelo puede generar respuestas imperfectas, asi que el proyecto debe defenderse explicando sus limites y no vendiendolo como una IA infalible.
- Si no existen pesos entrenados localmente, el backend usa un modelo preentrenado para asegurar una demo estable.
- CORS esta configurado para desarrollo local.

## Valor para el TFG

Este proyecto permite defender conocimientos reales adquiridos durante el ciclo:

- Estructura de proyecto full-stack.
- Separacion entre cliente y servidor.
- Consumo de APIs desde Angular.
- Validacion de datos con Pydantic.
- Uso basico de modelos de IA con Python.
- Control de dependencias con pnpm y uv.
- Diseno de una experiencia de usuario coherente.
- Capacidad de explicar decisiones, limitaciones y mejoras futuras.
