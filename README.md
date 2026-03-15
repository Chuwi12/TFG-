# Chatbot IA - Trabajo de Fin de Grado

Este repositorio contiene el código fuente de nuestro chatbot, dividido en un frontend hecho con Angular y un backend de Inteligencia Artificial construido con Python y PyTorch.

## Requisitos Previos
Asegúrate de tener instalado en tu computadora:
- [Node.js](https://nodejs.org/) y **pnpm** (puedes instalarlo con `npm install -g pnpm`)
- [Python 3.8+](https://www.python.org/) y **uv** (gestor ultrarrápido para Python)
- Git

---

## Instalación y Configuración (Primera vez)

Sigue estos pasos en orden para levantar el proyecto en tu entorno local:

## Activar frontend
```bash
cd frontend
pnpm install
pnpm start
```
## Activar backend
```bash
cd backend
```

### Creación entorno virtual backend
```bash
uv venv
```

### Instalación de librerías necesarias
```bash
uv pip install -r requirements.txt
```

#### Activación de entorno virtual backend
Si se usa shell fish
```bash
source .venv/bin/activate.fish
```
Si se usa bash o zsh
```bash
source .venv/bin/activate
```

#### Desactivación de entorno virtual backend
```bash
deactivate
```

