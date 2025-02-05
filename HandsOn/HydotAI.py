import torch

# Load the complete trained model
model = torch.load('trained_model_full.pth')
model.eval()  # Set the model to evaluation mode

# Chatbot function
def chat():
    print("Hello! I am your chatbot. Ask me anything, and I will try to answer.")
    while True:
        question = input("You: ")
        
        # Exit condition
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        # Generate the answer using the trained model
        answer = model.generate_answer(question, max_new_tokens=200)
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    chat()
