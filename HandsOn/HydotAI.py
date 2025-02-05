import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotModel(torch.nn.Module):
    def __init__(self):
        super(ChatbotModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def forward(self, input_ids):
        return self.model(input_ids)

def load_model(model_path):
    model = ChatbotModel()
    model.load_state_dict(torch.load(model_path), strict=False)  # Allow mismatched keys
    model.eval()
    return model

def preprocess_input(user_input, tokenizer):
    inputs = tokenizer(user_input, return_tensors="pt")
    return inputs['input_ids']

def generate_response(model, processed_input):
    with torch.no_grad():
        output = model(processed_input)
        # Decode the generated tokens into text
        response = model.tokenizer.decode(output.logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)
    return response

def chatbot():
    print("Hello! Ask me anything, and I will try to answer.")
    
    model_path = "trained_model.pth"
    model = load_model(model_path)
    
    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        processed_input = preprocess_input(user_input, model.tokenizer)
        response = generate_response(model, processed_input)
        
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
