import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import json

# Importamos nuestro modelo y nuestro dataset de los otros archivos
from model import NeuralNetwork
from dataset import BankingDataset

class ModelTrainer:
    def __init__(self, batch_size=32, learning_rate=0.001, epochs=20, load_path=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Usar la tarjeta gráfica (GPU) si está disponible, sino usar el procesador (CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = NeuralNetwork().to(self.device)

        # Cargar pesos si se especifica un modelo previo y el archivo existe
        if load_path and os.path.exists(load_path):
            print(f"Cargando pesos preentrenados desde: {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))

        # CrossEntropyLoss es la función de pérdida estándar para clasificación multiclase (77 clases)
        self.criterion = nn.CrossEntropyLoss()
        # El optimizador Adam es muy eficiente para ajustar los pesos de la red
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("Descargando/Cargando dataset...")
        hf_dataset = load_dataset("mteb/banking77")
        
        # Guardar el mapa de etiquetas automáticamente
        if 'label' in hf_dataset['train'].features:
            label_names = hf_dataset['train'].features['label'].names
            labels_map = {str(i): name for i, name in enumerate(label_names)}
            with open("labels.json", "w") as f:
                json.dump(labels_map, f, indent=4)
            print("Etiquetas guardadas automáticamente en labels.json")
            
        print("Preparando el conjunto de entrenamiento...")
        train_dataset = BankingDataset(hf_dataset['train'])
        # DataLoader se encarga de agrupar los datos en "batches" (lotes) y mezclarlos
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Iniciando entrenamiento en: {self.device}")
        self.model.train() # Ponemos el modelo en modo entrenamiento
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                # Movemos los datos a la CPU o GPU
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 1. Hacia adelante (Forward Pass)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # 2. Hacia atrás (Backward Pass) y optimización
                self.optimizer.zero_grad() # Limpiamos gradientes anteriores
                loss.backward()            # Calculamos los nuevos gradientes
                self.optimizer.step()      # Actualizamos los pesos de la neurona
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}] - Pérdida media: {avg_loss:.4f}")
        
        print("¡Entrenamiento finalizado!")
        self.save_model()

    def save_model(self, filepath="intent_classifier.pth"):
        """Guarda el estado del modelo para poder usarlo después en la API sin volver a entrenar."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Modelo guardado exitosamente en: {filepath}")

if __name__ == "__main__":
    # Si quieres empezar de cero, deja load_path=None
    # Si quieres re-entrenar, cambia a: load_path="intent_classifier.pth"
    trainer = ModelTrainer(epochs=20, load_path=None)
    trainer.train()
