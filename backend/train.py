import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ChatDataset
from model import CausalTransformer


class ModelTrainer:
    def __init__(
        self,
        vocab_model_name="datificate/gpt2-small-spanish",
        batch_size=2,
        learning_rate=5e-5,
        epochs=1,
        dataset_limit=500
    ):
        self.vocab_model_name = vocab_model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_limit = dataset_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Cargando tokenizador de: {self.vocab_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.vocab_model_name)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
            "pad_token": "<pad>"
        })

        self.model = CausalTransformer(vocab_size=len(self.tokenizer)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        dataset = ChatDataset(
            tokenizer=self.tokenizer,
            max_length=512,
            split="train",
            limit=self.dataset_limit
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"Iniciando entrenamiento en: {self.device}")
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                _, loss = self.model(input_ids, labels=labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"Epoch [{epoch + 1}/{self.epochs}] - perdida media: {avg_loss:.4f}")

        self.save_model()

    def save_model(self):
        save_dir = "./saved_chat_model"
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "custom_model.pth")
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Modelo y tokenizador guardados en {save_dir}")


if __name__ == "__main__":
    trainer = ModelTrainer(epochs=1, batch_size=2, dataset_limit=500)
    trainer.train()
