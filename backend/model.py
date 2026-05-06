import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


class CausalTransformer(nn.Module):
    """Transformer causal sencillo usado para explicar el ciclo de entrenamiento."""

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        seq_len = min(input_ids.size(1), self.max_seq_len)
        input_ids = input_ids[:, :seq_len]

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(x)

        loss = None
        if labels is not None:
            labels = labels[:, :seq_len]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss


class ChatModel:
    def __init__(self, vocab_model_name="datificate/gpt2-small-spanish", device=None, load_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "pretrained"
        
        print(f"Cargando tokenizador de: {vocab_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_model_name)
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
            "pad_token": "<pad>"
        })
        
        if load_path and os.path.exists(load_path):
            print(f"Cargando CausalTransformer entrenado desde {load_path}")
            self.model = CausalTransformer(vocab_size=len(self.tokenizer))
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            self.model_type = "custom"
        else:
            print(f"Cargando modelo pre-entrenado {vocab_model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                vocab_model_name,
                pad_token_id=self.tokenizer.pad_token_id
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, user_text, max_new_tokens=100):
        input_text = f"<|prompter|>{user_text}</s><|assistant|>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.model_type == "custom":
                output = self._generate_with_custom_model(input_ids, max_new_tokens)
            else:
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

        full_response = self.tokenizer.decode(output[0], skip_special_tokens=False)

        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].replace("</s>", "").replace("<pad>", "").strip()
        else:
            response = full_response.replace("</s>", "").replace("<pad>", "").strip()
            
        return response

    def _generate_with_custom_model(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.model.max_seq_len:
                break

            logits, _ = self.model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            token_id = next_token.item()
            if token_id in {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}:
                break

        return input_ids
