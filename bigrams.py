# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 20000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# read the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

length = len(text)
print("length of the dataset - ", length)

# build out the vocabulary of the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars) # size = 65

# create encode decode functions
stoi = {s : i for i,s in enumerate(chars)}
itos = {i : s for s,i in stoi.items()}

def encode(s):
    out = []
    for i in s:
        out.append(stoi[i])
    return out
def decode(s):
    out = ""
    for i in s:
        out+=itos[i]
    return out

# preparing the data 
data = torch.tensor(encode(text), dtype=torch.long)

# split into train and val
limit = int(0.9 * len(data))
Xtr = data[:limit] # (torch.Size([1003854])
Xval = data[limit:] # torch.Size([111540])

def get_batch(split):
    data = Xtr if split=='train' else Xval
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# the fun part
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):   
        logits = self.token_embedding_table(idx) #(B,T,C)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # pytorch expects this dim
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        
m = BigramLanguageModel(vocab_size).to(device)

# optimiser
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for it in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("===========================OUTPUT===========================")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
