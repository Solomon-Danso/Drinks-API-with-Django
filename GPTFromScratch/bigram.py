import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
# Tokenization 
stoi = {}
for index, character in enumerate(chars):
    stoi[character] = index

# Step 3: Create a dictionary to map each number back to a character
itos = {}
for index, character in enumerate(chars):
    itos[index] = character

# Step 4: Define the encoder function
def encode(s):
    result = []  # Empty list to store the indices
    for c in s:  # Loop through each character in the string
        result.append(stoi[c])  # Add the corresponding index to the list
    return result

# Step 4: Define the decoder function using a loop
def decode(l):
    result = ""  # Empty string to store the decoded characters
    for i in l:  # Loop through each index in the list
        result += itos[i]  # Add the corresponding character to the string
    return result


# The encode and decode process is just finding the index in chars, basically !"#$%&'()*+,-./3:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}

msg = "Hydot Tech Is Back"
msgIndex = encode(msg)
print("Encoded Message: ",msgIndex)

theMsgIndex = msgIndex
theDecodeMsg = decode(theMsgIndex)
print("Decoded Message ",theDecodeMsg)


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    """
    Generates a batch of input sequences (x) and target sequences (y).

    - 'split' determines whether we use training data ('train') or validation data ('val').
    - Each sequence in x has block_size elements.
    - Each sequence in y is shifted by 1 position (so the model learns to predict the next value).
    """

    # Select the appropriate dataset based on the split ('train' or 'val')
    data = train_data if split == 'train' else val_data  

    # Generate random starting indices for batch sequences
    # Ensures that we pick sequences without exceeding data length
    start_indices = torch.randint(len(data) - block_size, (batch_size,))
    print("Start Indices:", start_indices)

    # Create input sequences (x) by slicing data from the selected indices
    # Each row in x contains block_size consecutive values from data
    x = torch.stack([data[i:i+block_size] for i in start_indices])
    print("The X (Input Sequences):", x)

    # Create target sequences (y) by shifting each input sequence by one position
    # This helps the model learn to predict the next value in the sequence
    y = torch.stack([data[i+1:i+block_size+1] for i in start_indices])
    print("The Y (Target Sequences):", y)

    return x, y  # Return the input (x) and target (y) sequences


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
