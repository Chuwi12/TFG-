import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

class ChatDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, max_length=512, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Cargando OpenAssistant/oasst1 ({split})... esto puede tardar un poco.")
        ds = load_dataset('OpenAssistant/oasst1')
        df = ds[split].to_pandas()
        
        # Filtrar por idioma español
        df_es = df[df['lang'] == 'es']
        
        # Separar roles
        prompters = df_es[df_es['role'] == 'prompter']
        assistants = df_es[df_es['role'] == 'assistant']
        
        # Unir para crear pares pregunta-respuesta
        pairs = pd.merge(prompters, assistants, left_on='message_id', right_on='parent_id', suffixes=('_p', '_a'))
        
        self.texts = []
        for _, row in pairs.iterrows():
            prompt = row['text_p']
            response = row['text_a']
            # Formato conversacional
            text = f"<|prompter|>{prompt}</s><|assistant|>{response}</s>"
            self.texts.append(text)
            
        print(f"Dataset listo: {len(self.texts)} conversaciones en español encontradas.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Para modelos causales, los labels son los mismos input_ids
        # Ignoramos el padding en la pérdida poniendo -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
