import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
block_size = 64
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 64
num_heads = 4
num_layers = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open("data/tinyshakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
def encode(s):
    # E371:
    return [stoi[chr] for chr in s]  # encoder: take a string, output a list of integers


# decode = lambda l: ''.join([itos[i] for i in l]) # decoder:
def decode(ints: list[int]) -> str:
    # E371:
    return "".join(
        [itos[i] for i in ints]
    )  # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

def get_batch(split, device=device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)

        # mask out the future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(weights, dim=-1)  # (B,T,T)
        weights = self.dropout(weights)
        # weights = self.dropout(weights)

        v = self.value(x)  # (B,T,head_size)

        out = weights @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class LayerNorm:
    def __init__(self,  dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B,T,Cemb)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # in bigram model we don't use positional embeddings 
        x = token_embeddings + pos_embeddings  # (B,T,Cemb)
        x = self.blocks(x)  # (B,T,Cemb)
        logits = self.lm_head(x)  # (B,T,Cvocab)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to the last block_size tokens
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


if __name__ == "__main__":
    # create the bigram model instance
    model = BigramLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))