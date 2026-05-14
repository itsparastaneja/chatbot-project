import streamlit as st
import torch
import torch.nn as nn
import pickle
import os

st.set_page_config(page_title="Chatbot", page_icon="🤖")

#  LOAD MODEL 
@st.cache_resource
def load_model():
    # Load vocab
    with open("models/vocab.pkl", "rb") as f:
        data = pickle.load(f)
    
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    VOCAB_SIZE = len(word2idx)
    
    # Model definition (same as train.py)
    class SimpleChatbot(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, 64)
            self.encoder = nn.LSTM(64, 128, batch_first=True)
            self.decoder = nn.LSTM(64, 128, batch_first=True)
            self.fc = nn.Linear(128, VOCAB_SIZE)
        
        def reply(self, src, max_len=8):
            with torch.no_grad():
                _, (h, c) = self.encoder(self.embedding(src.unsqueeze(0)))
                
                token = torch.tensor([[1]])
                result = []
                
                for _ in range(max_len):
                    emb = self.embedding(token)
                    out, (h, c) = self.decoder(emb, (h, c))
                    pred = self.fc(out).argmax(-1).item()
                    
                    if pred == 2:
                        break
                    if pred > 3:
                        result.append(pred)
                    
                    token = torch.tensor([[pred]])
                
                return result
    
    # Load model weights
    model = SimpleChatbot()
    model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    model.eval()
    
    return model, word2idx, idx2word

#  TEXT TO TENSOR 
def text_to_tensor(text):
    model, word2idx, idx2word = load_model()
    
    tokens = text.lower().split()[:7]
    ids = [word2idx.get(w, 3) for w in tokens] + [2]  # +EOS
    
    if len(ids) < 8:
        ids = ids + [0] * (8 - len(ids))
    else:
        ids = ids[:8]
    
    return torch.tensor(ids, dtype=torch.long)

# ========== UI ==========
st.title("🤖 Chatbot")
st.caption("A simple AI chatbot")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Type something!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Type here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    try:
        model, word2idx, idx2word = load_model()
        src = text_to_tensor(prompt)
        result_ids = model.reply(src)
        response = " ".join([idx2word.get(i, "") for i in result_ids])
        
        if not response:
            response = "I don't understand. Try: hello, how are you, bye"
    
    except Exception as e:
        response = "Error! Did you run train.py first?"
    
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.write(response)

# Sidebar
with st.sidebar:
    st.markdown("### Try these words:")
    st.markdown("- hello\n- hi\n- how are you\n- what is your name\n- thank you\n- bye")
    
    if st.button("🔄 Restart Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Type something!"}
        ]
        st.rerun()