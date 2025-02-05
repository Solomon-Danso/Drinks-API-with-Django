import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# Hyperparameters
batch_size = 16
block_size = 128  # Larger context for question-answering
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128  # Larger embedding size for better performance in Q&A
n_head = 8
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)

# Load your notes or text data
with open('/Users/glydetek/Desktop/Hydot/Contributions/Learn Django/Data/source.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Store vocab for later use
vocab = {'stoi': stoi, 'itos': itos}
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)




# Transformer-based model classes (Head, MultiHeadAttention, FeedForward, Block, TransformerQAModel)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_answer(self, question, max_new_tokens=200):
        """
        Generates a human-readable answer based on the input question.

        Args:
            question (str): The input question.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated answer in human-readable form.
        """
        # Tokenize the input question
        idx = torch.tensor(encode(question), dtype=torch.long, device=device).unsqueeze(0)

        generated_tokens = []
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Keep the last block_size tokens
            logits, _ = self(idx_cond)  # Get model predictions
            logits = logits[:, -1, :]  # Take the last token logits
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next.item())

            # Decode the response
            answer = decode(generated_tokens)

            # Stop generating if we hit a sentence boundary
            if answer.strip()[-1] in ['.', '!', '?']:
                break

        # Clean up unwanted characters
        answer = answer.replace("\n", " ").strip()

        # Remove any potential tokenization artifacts
        answer = ''.join(c for c in answer if c.isprintable())

        return answer 
    
    
model = TransformerQAModel().to(device)

save_path = "trained_model_full.pth"
torch.save(model, save_path)
print(f"Trained model saved to {save_path}")

# Start the chatbot interaction
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == 'exit':
        break
    answer = model.generate_answer(question, max_new_tokens=200)
    print(f"Chatbot: {answer}")
