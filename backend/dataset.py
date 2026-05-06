from torch.utils.data import Dataset
from datasets import load_dataset


class ChatDataset(Dataset):
    """Prepara pares pregunta-respuesta en espanol para entrenar el chatbot."""

    def __init__(self, tokenizer, max_length=512, split="train", limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []

        print(f"Cargando OpenAssistant/oasst1 ({split})...")
        dataset = load_dataset("OpenAssistant/oasst1", split=split)

        spanish_messages = [row for row in dataset if row.get("lang") == "es"]
        assistants_by_parent = {
            row["parent_id"]: row["text"]
            for row in spanish_messages
            if row.get("role") == "assistant" and row.get("parent_id")
        }

        for row in spanish_messages:
            if row.get("role") != "prompter":
                continue

            response = assistants_by_parent.get(row["message_id"])
            if not response:
                continue

            text = f"<|prompter|>{row['text']}</s><|assistant|>{response}</s>"
            self.texts.append(text)

            if limit and len(self.texts) >= limit:
                break

        print(f"Dataset listo: {len(self.texts)} conversaciones en espanol.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
