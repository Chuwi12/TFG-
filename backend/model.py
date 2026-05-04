import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
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
        
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        mask = self.generate_square_subsequent_mask(seq_len, input_ids.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(x)
        
        loss = None
        if labels is not None:
            # Desplazamos los logits y labels para predecir el siguiente token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss

class ChatModel:
    def __init__(self, vocab_model_name="datificate/gpt2-small-spanish", device=None, load_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Cargando tokenizador de: {vocab_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_model_name)
        
        # Añadir tokens especiales
        special_tokens = {
            "additional_special_tokens": ["<|prompter|>", "<|assistant|>"],
            "pad_token": "<pad>"
        }
        self.tokenizer.add_special_tokens(special_tokens)
        vocab_size = len(self.tokenizer)
        
        print("Inicializando red neuronal propia desde cero...")
        # Instanciar nuestro modelo Transformer propio
        self.model = CausalTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(self.device)
        
        if load_path and os.path.exists(load_path):
            print(f"Cargando pesos pre-entrenados desde {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        
    def generate_response(self, user_text, max_length=50):
        self.model.eval()
        input_text = f"<|prompter|>{user_text}</s><|assistant|>"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_ids.size(1)
                if seq_len >= self.model.max_seq_len:
                    break
                    
                logits, _ = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # Búsqueda greedy (cogemos el token más probable)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == self.tokenizer.pad_token_id:
                    break
                    
        full_response = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].replace("</s>", "").replace("<pad>", "").strip()
        else:
            response = full_response.strip()
            
        return response
