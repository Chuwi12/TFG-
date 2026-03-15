from datasets import load_dataset

print("Conectando con Hugging Face y descargando Banking77 limpio...")

dataset = load_dataset("mteb/banking77")

print("\n¡Descarga completada!")
print("--- Ejemplo del primer mensaje de entrenamiento ---")
print("Texto del usuario:", dataset['train'][0]['text'])
print("Etiqueta (ID de la intención):", dataset['train'][0]['label'])

print(f"\nTotal de frases listas para entrenar: {len(dataset['train'])}")