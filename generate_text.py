# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pickle
import random

class CharModel(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim=256,
                 hidden_size=512,
                 num_layers=5,
                 dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

class TextGenerator:
    def __init__(self, model_path='saved_models/final_model(loss= 0.45 DIC = 1500, 5 layers V1.4).pth', vocab_path='training_data/vocab.pkl', debug_mode=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug_mode = debug_mode
        print(f"Using device: {self.device}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.model = CharModel(
            vocab_size=len(checkpoint['combined_vocab']),
            embedding_dim=256,
            hidden_size=512,
            num_layers=5,
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.combined_vocab = checkpoint['combined_vocab']
        self.idx2combined = checkpoint['idx2combined']
        
        with open(vocab_path, 'rb') as f:
            self.word_vocab = pickle.load(f)

    def generate(self, seed="", length=1000, temp=0.8, stream=False):
        with torch.no_grad():
            generated = []
            current_seq = []
            
            seed_tokens = seed.split()
            
            for i, token in enumerate(seed_tokens):
                if token in self.word_vocab:
                    idx = self.word_vocab[token]
                else:
                    idx = self.combined_vocab.get(token, 
                            random.choice(list(self.combined_vocab.values())))
                
                current_seq.append(idx)
                
                if self.debug_mode:
                    formatted_token = f"[{token}]"
                else:
                    formatted_token = token
                
                if i < len(seed_tokens) - 1:
                    formatted_token += ' '
                
                generated.append(formatted_token)
                if stream:
                    print(formatted_token, end='', flush=True)
            
            hidden = None
            
            for _ in range(length):
                x = torch.tensor(current_seq[-100:], dtype=torch.long).unsqueeze(0).to(self.device)
                output, hidden = self.model(x, hidden)
                
                last_logits = output[0, -1] / temp
                probs = torch.softmax(last_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                current_seq.append(next_idx)
                
                token = self.idx2combined[next_idx]
                if self.debug_mode:
                    formatted_token = f"[{token}]"
                else:
                    formatted_token = token
                
                generated.append(formatted_token)
                if stream:
                    print(formatted_token, end='', flush=True)
            
            if stream:
                print()
            return ''.join(generated)

if __name__ == '__main__':
    debug_input = input("To enter debug mode, write 'D', otherwise press enter: ")
    debug_mode = debug_input.strip().upper() == 'D'
    
    generator = TextGenerator(debug_mode=debug_mode)
    while True:
        seed = input("\nEnter starting text (or 'quit' to exit): ")
        if seed.lower() == 'quit':
            break
        print("\nGenerated text:")
        generator.generate(seed=seed, stream=True)
        print("\n" + "-"*50)