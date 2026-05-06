import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer

# Importamos nuestro modelo y nuestro dataset de los otros archivos
from model import CausalTransformer
from dataset import ChatDataset

class ModelTrainer:
    def __init__(self, vocab_model_name="datificate/gpt2-small-spanish", batch_size=4, learning_rate=5e-5, epochs=10):
        self.vocab_model_name = vocab_model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Cargando tokenizador de: {self.vocab_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.vocab_model_name)
        special_tokens = {
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
            "pad_token": "<pad>"
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        vocab_size = len(self.tokenizer)
        
        # Construimos la red neuronal y el optimizador antes de entrenar
        self.build_model(vocab_size)

    def build_model(self, vocab_size):
        self.model = CausalTransformer(vocab_size=vocab_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("Preparando el conjunto de entrenamiento (ChatDataset)...")
        # Instanciamos nuestra clase ChatDataset que descarga OpenAssistant y filtra por español
        train_dataset = ChatDataset(self.tokenizer, max_length=512, split="train")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Iniciando entrenamiento en: {self.device}")
        self.model.train() # Ponemos el modelo en modo entrenamiento
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Movemos los tensores a la CPU/GPU
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 1. Hacia adelante (Forward Pass)
                logits, loss = self.model(input_ids, labels=labels)
                
                # 2. Hacia atrás (Backward Pass) y optimización
                self.optimizer.zero_grad() # Limpiamos gradientes anteriores
                loss.backward()            # Calculamos los nuevos gradientes
                self.optimizer.step()      # Actualizamos los pesos de la neurona
                
                running_loss += loss.item()
                
                # Imprimir el progreso cada 50 lotes
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - Lote [{batch_idx+1}/{len(train_loader)}] - Pérdida: {loss.item():.4f}")
            
            avg_loss = running_loss / len(train_loader)
            print(f"--- Fin de Epoch [{epoch+1}/{self.epochs}] - Pérdida media: {avg_loss:.4f} ---")
        
        print("¡Entrenamiento finalizado para el chatbot!")
        self.save_model()

    def save_model(self):
        """Guarda el estado del modelo para poder usarlo después en la API sin volver a entrenar."""
        save_dir = "./saved_chat_model"
        # Ajustamos la ruta para que se guarde en la raíz del proyecto, como espera main.py
        # ya que train.py normalmente se ejecutará desde el directorio backend o la raíz.
        if os.path.basename(os.getcwd()) == 'backend':
            save_dir = "../saved_chat_model"
            
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "custom_model.pth")
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Modelo y tokenizador guardados exitosamente en {save_dir}")

if __name__ == "__main__":
    # Instanciar y entrenar el ModelTrainer para el chatbot
    # Se usan 3 épocas y un batch_size de 4 por defecto (ajustable según la VRAM disponible)
    trainer = ModelTrainer(vocab_model_name="datificate/gpt2-small-spanish", epochs=3, batch_size=4)
    trainer.train()
