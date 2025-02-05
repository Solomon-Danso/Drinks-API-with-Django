import torch
import json
import numpy as np
import torch.nn.functional as F
from HydotLLM2 import TransformerQAModel

# Load vocab
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

stoi = vocab['stoi']  # String to index
itos = vocab['itos']  # Index to string

# Load the trained model
model = torch.load('/Users/glydetek/Desktop/Hydot/Contributions/Learn Django/HandsOn/trained_model_full.pth')
model.eval()  # Set the model to evaluation mode

# Function to process input query
def process_input(query):
    tokens = [stoi.get(char, stoi[' ']) for char in query]  # Map chars to indices
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

# Function to process output from model
def process_output(output):
    output = output.squeeze(0)  # Remove batch dimension
    output_tokens = output.argmax(dim=-1).cpu().numpy()  # Get token with highest probability
    response = ''.join([itos[token] for token in output_tokens if token != stoi[' ']])  # Convert tokens to string
    return response

# Function for chatbot response
def chatbot():
    print("Chatbot: Hello! Ask me anything...")
    while True:
        query = input("You: ")  # Get user input
        if query.lower() == 'exit':  # End chat on 'exit'
            print("Chatbot: Goodbye!")
            break
        
        # Process the input query
        input_tensor = process_input(query)
        
        # Pass the input through the model
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process and print the model's response
        response = process_output(output)
        print(f"Chatbot: {response}")

# Run chatbot
if __name__ == "__main__":
    chatbot()
