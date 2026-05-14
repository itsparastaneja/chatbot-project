# 🤖 HelpDesk AI Chatbot

## Project Overview
AI-powered chatbot using Seq2Seq deep learning model with LSTM neural networks. 
Can understand user input and generate human-like responses.

## Technology Used
- Python 3.x
- PyTorch (Deep Learning Framework)
- Streamlit (Web Interface)
- LSTM Neural Networks

## Installation & Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Train the AI model:
   python train.py

3. Launch chatbot web app:
   python -m streamlit run app.py

4. Open browser at: http://localhost:8501/

## Project Structure
├── train.py          (Model training)
├── app.py            (Web interface)
├── requirements.txt  (Dependencies)
├── models/           (Trained model files)
└── README.md         (This file)

## How It Works
- User types a message
- Text is converted to numbers
- Neural network processes it
- AI generates a response
- Response shown in chat interface

## Note
This is an AI/ML project that runs locally.
Requires Python and model training before use.

