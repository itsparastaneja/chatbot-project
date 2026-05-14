import torch
import torch.nn as nn
import pickle
import random
import os

print("="*50)
print("TRAINING CHATBOT - PLEASE WAIT 5 MINUTES")
print("="*50)

#  SIMPLE SETTINGS 
MAX_LEN = 8

#  TRAINING DATA
conversations = [
    ("hello", "hi there"),
    ("hi", "hello"),
    ("hey", "hey there"),
    ("how are you", "i am good"),
    ("how are you", "i am fine thanks"),
    ("what is your name", "my name is bot"),
    ("who are you", "i am a chatbot"),
    ("what can you do", "i can chat"),
    ("help me", "how can i help"),
    ("thank you", "you are welcome"),
    ("thanks", "no problem"),
    ("bye", "goodbye"),
    ("goodbye", "see you"),
    ("good morning", "morning"),
    ("good night", "night"),
    ("i am happy", "that is good"),
    ("i am sad", "do not be sad"),
    ("you are smart", "thank you"),
    ("you are funny", "thanks"),
    ("tell me a joke", "i am not funny"),
]

#  BUILD VOCABULARY 
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
word2idx = {PAD:0, SOS:1, EOS:2, UNK:3}
idx2word = {0:PAD, 1:SOS, 2:EOS, 3:UNK}

counter = 4
for q, a in conversations:
    for w in q.split() + a.split():
        if w not in word2idx:
            word2idx[w] = counter
            idx2word[counter] = w
            counter += 1

VOCAB_SIZE = len(word2idx)
print(f"Vocabulary: {VOCAB_SIZE} words")

# Save vocabulary
os.makedirs("models", exist_ok=True)
with open("models/vocab.pkl", "wb") as f:
    pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)

# CREATE DATASET 
def text_to_tensor(text, add_sos=False):
    if add_sos:
        tokens = [SOS] + text.split() + [EOS]
    else:
        tokens = text.split() + [EOS]
    
    ids = [word2idx.get(w, UNK) for w in tokens]
    
    # Pad or truncate
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids = ids + [0] * (MAX_LEN - len(ids))
    
    return torch.tensor(ids, dtype=torch.long)

# Make training data (repeat conversations)
all_pairs = []
for _ in range(20):
    all_pairs.extend(conversations)
random.shuffle(all_pairs)

print(f"Training samples: {len(all_pairs)}")

#  SIMPLE MODEL 
class SimpleChatbot(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.encoder = nn.LSTM(64, 128, batch_first=True)
        self.decoder = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, VOCAB_SIZE)
    
    def forward(self, src, tgt):
        # Encode
        _, (h, c) = self.encoder(self.embedding(src))
        # Decode
        tgt_emb = self.embedding(tgt[:, :-1])
        out, _ = self.decoder(tgt_emb, (h, c))
        return self.fc(out)
    
    def reply(self, src, max_len=8):
        with torch.no_grad():
            _, (h, c) = self.encoder(self.embedding(src.unsqueeze(0)))
            
            token = torch.tensor([[1]])  # SOS
            result = []
            
            for _ in range(max_len):
                emb = self.embedding(token)
                out, (h, c) = self.decoder(emb, (h, c))
                pred = self.fc(out).argmax(-1).item()
                
                if pred == 2:  # EOS
                    break
                if pred > 3:
                    result.append(pred)
                
                token = torch.tensor([[pred]])
            
            return result

#  TRAIN 
model = SimpleChatbot()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=0)

EPOCHS = 50
best_loss = 999

print("\nTraining...")
for epoch in range(EPOCHS):
    total_loss = 0
    count = 0
    
    for q, a in all_pairs[:100]:
        src_tensor = text_to_tensor(q)
        tgt_tensor = text_to_tensor(a, add_sos=True)
        
        # Forward
        output = model(src_tensor.unsqueeze(0), tgt_tensor.unsqueeze(0))
        output = output.view(-1, VOCAB_SIZE)
        target = tgt_tensor[1:MAX_LEN]  # Remove SOS
        
        loss = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    avg_loss = total_loss / count
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "models/model.pt")
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print(f"\nTraining done! Best loss: {best_loss:.4f}")
torch.save(model.state_dict(), "models/model.pt")

# ========== TEST ==========
print("\n" + "="*50)
print("TESTING CHATBOT:")
print("="*50)

model.eval()
tests = ["hello", "how are you", "what is your name", "bye", "thank you"]

for t in tests:
    src = text_to_tensor(t)
    ids = model.reply(src)
    response = " ".join([idx2word.get(i, "") for i in ids])
    
    print(f"👤 {t}")
    print(f"🤖 {response}\n")

print("="*50)
print("✅ DONE! Run: python -m streamlit run app.py")
print("="*50)