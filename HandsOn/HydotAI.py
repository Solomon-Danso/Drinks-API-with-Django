import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

labels = {"greeting": 0, "farewell": 1}  # Ensure this matches training labels

# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x.float())

# Load trained model
input_size = len(vocab)
output_size = len(labels)
model = SimpleNN(input_size, output_size)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Prediction function
def predict(text):
    tokens = text.split()
    vector = torch.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            vector[vocab[token]] = 1
    output = model(vector.unsqueeze(0))
    prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return [key for key, value in labels.items() if value == prediction][0]

# Chatbot Interaction
def chat():
    print("HydotAI: Hello! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("HydotAI: Goodbye!")
            break
        response = predict(user_input)
        print(f"HydotAI: {response}")

if __name__ == "__main__":
    chat()
