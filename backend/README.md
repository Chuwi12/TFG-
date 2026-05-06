# Backend - THE TICKER

Este backend expone una API sencilla con FastAPI para que el frontend pueda enviar preguntas al chatbot.

## Enfoque Actual

El backend tiene dos partes complementarias:

1. **Pipeline educativo de entrenamiento:** `dataset.py`, `train.py` y `CausalTransformer` muestran como se prepara un dataset conversacional, se entrena un modelo causal sencillo y se guardan pesos locales.
2. **Ruta estable de demo:** si no existe `saved_chat_model/custom_model.pth`, la API usa un modelo de lenguaje preentrenado en espanol (`datificate/gpt2-small-spanish`) mediante `transformers`.

Esta decision mantiene el proyecto simple y defendible: se puede explicar el ciclo de IA sin depender de que el entrenamiento salga perfecto durante la presentacion.

## Archivos Principales

- `main.py`: define la API, CORS y el endpoint `/chat`.
- `model.py`: contiene `CausalTransformer`, carga el modelo disponible y genera respuestas.
- `dataset.py`: prepara conversaciones en espanol de OpenAssistant.
- `train.py`: entrena el `CausalTransformer` y guarda `saved_chat_model/custom_model.pth`.
- `pyproject.toml`: declara las dependencias necesarias del backend.

## Estrategia de Demo

Para grabar o defender el proyecto:

- Si existe `saved_chat_model/custom_model.pth`, el backend carga el `CausalTransformer` entrenado.
- Si no existe, el backend carga el modelo preentrenado en espanol y la API sigue funcionando.

Asi el proyecto no queda bloqueado por el coste de entrenamiento, pero conserva una parte de IA propia para explicar el aprendizaje tecnico.

## Entrenar Opcionalmente

```bash
uv run python train.py
```

El entrenamiento usa un limite pequeno de ejemplos para que sea razonable en un entorno de clase. Se puede aumentar `dataset_limit`, `epochs` o `batch_size` si hay mas tiempo o mejor hardware.

## Ejecutar

```bash
uv venv
uv sync
uv run python main.py
```

La API queda disponible en:

```text
http://localhost:8000
```
