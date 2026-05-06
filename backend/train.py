import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from transformers import AutoTokenizer
import copy

# Importamos nuestro modelo y nuestro dataset de los otros archivos
from model import CausalTransformer
from dataset import ChatDataset

class ModelTrainer:
    def __init__(self, vocab_model_name="datificate/gpt2-small-spanish", batch_size=4, learning_rate=3e-4, epochs=10, accumulation_steps=4):
        self.vocab_model_name = vocab_model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Acumulación de gradientes para simular un batch size mayor (batch_size * accumulation_steps)
        self.accumulation_steps = accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Cargando tokenizador de: {self.vocab_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.vocab_model_name)
        special_tokens = {
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
            "pad_token": "<pad>"
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        vocab_size = len(self.tokenizer)
        
        # Construimos la red neuronal, optimizador y scaler para precisión mixta
        self.build_model(vocab_size)

    def build_model(self, vocab_size):
        self.model = CausalTransformer(vocab_size=vocab_size).to(self.device)
        # Usamos AdamW (con weight decay) que es mejor para Transformers que Adam normal
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # Reduce el learning rate si la pérdida de validación deja de mejorar
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        # Scaler para Entrenamiento de Precisión Mixta (AMP) para ahorrar VRAM y acelerar
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

    def get_save_dir(self):
        save_dir = "./saved_chat_model"
        if os.path.basename(os.getcwd()) == 'backend':
            save_dir = "../saved_chat_model"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_model(self, is_best=False):
        """Guarda el modelo. Si es el mejor hasta ahora, lo guarda como custom_model.pth"""
        save_dir = self.get_save_dir()
        filename = "custom_model.pth" if is_best else "latest_model.pth"
        model_path = os.path.join(save_dir, filename)
        
        torch.save(self.model.state_dict(), model_path)
        if is_best:
            self.tokenizer.save_pretrained(save_dir)
            print(f"Nuevo mejor modelo guardado en {model_path}!")

    def train(self):
        print("Preparando el conjunto de entrenamiento y validación (ChatDataset)...")
        train_dataset = ChatDataset(self.tokenizer, max_length=512, split="train")
        val_dataset = ChatDataset(self.tokenizer, max_length=512, split="validation")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Iniciando entrenamiento en: {self.device} con AMP {'activado' if torch.cuda.is_available() else 'desactivado'}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Fase de Entrenamiento
            self.model.train()
            running_loss = 0.0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Autocast para precisión mixta (hace los cálculos en fp16 si es posible)
                with autocast(enabled=torch.cuda.is_available()):
                    logits, loss = self.model(input_ids, labels=labels)
                    # Dividimos la pérdida por los pasos de acumulación
                    loss = loss / self.accumulation_steps
                
                # Backward con Scaler
                self.scaler.scale(loss).backward()
                
                # Actualizar pesos solo después de acumular suficientes gradientes
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Clip de gradientes para evitar la explosión de gradientes
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Restauramos la pérdida para las estadísticas
                running_loss += (loss.item() * self.accumulation_steps)
                
                if (batch_idx + 1) % 50 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}/{self.epochs}] - Lote [{batch_idx+1}/{len(train_loader)}] - Pérdida: {(loss.item() * self.accumulation_steps):.4f} - LR: {current_lr:.1e}")
            
            avg_train_loss = running_loss / len(train_loader)
            
            # Fase de Validación
            print("Evaluando modelo...")
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    with autocast(enabled=torch.cuda.is_available()):
                        _, loss = self.model(input_ids, labels=labels)
                        
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"--- Fin de Epoch [{epoch+1}/{self.epochs}] ---")
            print(f"Pérdida Entrenamiento: {avg_train_loss:.4f} | Pérdida Validación: {avg_val_loss:.4f}")
            
            # Ajustamos el learning rate basado en la validación
            self.scheduler.step(avg_val_loss)
            
            # Guardamos el mejor modelo si la pérdida de validación mejora
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(is_best=True)
            else:
                self.save_model(is_best=False) # Guardamos también el último estado por si acaso
                
        print("¡Entrenamiento finalizado para el chatbot!")

if __name__ == "__main__":
    # Ajusta los hiperparámetros. 
    # Al tener acumulación de gradientes (accumulation_steps=4), 
    # un batch_size de 4 equivale a un batch_size real de 16 (4x4).
    trainer = ModelTrainer(
        vocab_model_name="datificate/gpt2-small-spanish", 
        epochs=5, 
        batch_size=4, 
        accumulation_steps=4,
        learning_rate=3e-4 # Learning rate inicial un poco más alto, común con AdamW y Warmup/Plateau
    )
    trainer.train()