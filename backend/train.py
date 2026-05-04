import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import joblib
import kagglehub
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
        special_tokens = {"additional_special_tokens": ["<|prompter|>", "<|assistant|>"], "pad_token": "<pad>"}
        self.tokenizer.add_special_tokens(special_tokens)
        
        vocab_size = len(self.tokenizer)
        self.model = None
        # CrossEntropyLoss es la función de pérdida estándar para clasificación multiclase
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self, vocab_size):
        self.model = CausalTransformer(vocab_size=vocab_size).to(self.device)
        # El optimizador Adam es muy eficiente para ajustar los pesos de la red
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("Descargando/Cargando dataset...")
        hf_dataset = load_dataset("mteb/banking77")
        
        print("Preparando el conjunto de entrenamiento...")
        train_dataset = BankingDataset(hf_dataset['train'])
        # DataLoader se encarga de agrupar los datos en "batches" (lotes) y mezclarlos
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Iniciando entrenamiento en: {self.device}")
        self.model.train() # Ponemos el modelo en modo entrenamiento
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in train_loader:
                # Movemos los datos a la CPU o GPU
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 1. Hacia adelante (Forward Pass)
                logits, loss = self.model(input_ids, labels=labels)
                
                # 2. Hacia atrás (Backward Pass) y optimización
                self.optimizer.zero_grad() # Limpiamos gradientes anteriores
                loss.backward()            # Calculamos los nuevos gradientes
                self.optimizer.step()      # Actualizamos los pesos de la neurona
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader) # type: ignore
            print(f"Epoch [{epoch+1}/{self.epochs}] - Pérdida media: {avg_loss:.4f}")
        
        print("¡Entrenamiento finalizado para el chatbot!")
        self.save_model()

    def save_model(self):
        """Guarda el estado del modelo para poder usarlo después en la API sin volver a entrenar."""
        save_dir = "./saved_chat_model"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "custom_model.pth")
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Modelo y tokenizador guardados exitosamente en {save_dir}")

def get_csv_from_kaggle_path(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    return None

if __name__ == "__main__":
    # Instanciar y entrenar el ModelTrainer para el chatbot
    # The 'vocab_model_name' argument is used here, which fixes the TypeError.
    trainer = ModelTrainer(vocab_model_name="datificate/gpt2-small-spanish", epochs=3, batch_size=4)
    trainer.train()
