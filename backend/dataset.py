import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class BankingDataset(Dataset):
    def __init__(self, hf_dataset, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'):
        """
        Inicializa el dataset transformando el texto crudo en vectores de 512 dimensiones.
        """
        print("Cargando el modelo de lenguaje... (esto puede tardar unos segundos la primera vez)")
        self.vectorizer = SentenceTransformer(model_name)
        
        print("Extrayendo textos y etiquetas...")
        texts = [item['text'] for item in hf_dataset]
        labels = [item['label'] for item in hf_dataset]
        
        print(f"Vectorizando {len(texts)} frases a 512 dimensiones... (por favor, espera)")
        # Convertimos todos los textos a la vez. El resultado es un array de numpy
        embeddings = self.vectorizer.encode(texts, show_progress_bar=True)
        
        # Guardamos los datos como tensores de PyTorch listos para el modelo
        self.x = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        print("¡Dataset listo para entrenar!")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]