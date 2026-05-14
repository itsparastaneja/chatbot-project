import torch
import os

# Small settings for quick testing
MAX_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device("cpu")

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Create model directory
os.makedirs("models", exist_ok=True)