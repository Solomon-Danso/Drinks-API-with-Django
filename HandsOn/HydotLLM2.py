import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Load and preprocess text data
with open('/content/source.txt', 'r', encoding='utf-8') as f:
    text_data = f.readlines()

# Tokenize the text into words
tokenized_data = [line.strip().split() for line in text_data if line.strip()]

# Create vocabulary
vocab = {word: idx for idx, word in enumerate(set(word for sentence in tokenized_data for word in sentence))}

# Save vocabulary
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("Vocabulary size:", len(vocab))

# Dataset class
class NotesDataset(Dataset):
    def __init__(self, tokenized_data, vocab):
        self.data = tokenized_data
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        token_ids = [self.vocab.get(word, -1) for word in tokens]  # Default to -1 if word is not in vocab
        # Ensure token_ids are valid (no -1 values)
        token_ids = [token_id for token_id in token_ids if token_id != -1]
        return torch.tensor(token_ids, dtype=torch.long)

# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)  # Use embeddings instead of one-hot
        self.fc = nn.Linear(embedding_dim, output_size)  # A fully connected layer for the output
    
    def forward(self, x):
        embedded = self.embedding(x)  # Lookup embedding for each token
        # Aggregate the embeddings (e.g., by averaging)
        x = embedded.mean(dim=1)
        return self.fc(x)

# Prepare dataset and dataloader
embedding_dim = 50  # Set embedding dimension
dataset = NotesDataset(tokenized_data, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

input_size = len(vocab)
output_size = len(tokenized_data)  # Use number of sentences as output size
model = SimpleNN(input_size, embedding_dim, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100
for epoch in range(epochs):
    for text in dataloader:
        optimizer.zero_grad()
        output = model(text)  # Direct input to model after embedding lookup
        labels = torch.tensor([idx for idx in range(len(text))])  # Dummy labels for unsupervised task
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "trained_model.pth")
print("Model trained and saved as trained_model.pth")
