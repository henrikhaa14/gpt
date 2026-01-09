# =============================================================================
# GPT Text Generation Script
# =============================================================================
# Generate text using a trained GPT model.
# =============================================================================

# !pip install tiktoken

import tiktoken
import torch
import torch.nn as nn

# =============================================================================
# CONFIGURATION - Must match training settings!
# =============================================================================

# Model path
MODEL_PATH = "models/gpt_model.pt"

# Model architecture (must match training!)
VOCAB_SIZE = 50257
EMBEDDING_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2
MAX_SEQ_LEN = 512

# Generation settings
MAX_NEW_TOKENS = 100    # Number of tokens to generate
TEMPERATURE = 0.7       # Creativity: 0.1=focused, 1.0=creative, >1=chaotic
TOP_K = 50              # Sample from top K tokens (None=all)

# =============================================================================
# MODEL ARCHITECTURE (same as training)
# =============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT language model with text generation."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim=4 * embed_dim)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Initial token IDs (1, T)
            max_new_tokens: How many tokens to generate
            temperature: Controls randomness (lower=deterministic)
            top_k: Only sample from top K tokens
            
        Returns:
            Generated token IDs including the prompt
        """
        self.eval()
        tokens = start_tokens.clone()
        
        for _ in range(max_new_tokens):
            # Crop to max context length
            context = tokens[:, -self.max_seq_len:]
            
            # Get next token probabilities
            logits = self(context)[:, -1, :] / temperature
            
            # Optional: top-k filtering
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens


# =============================================================================
# LOAD MODEL
# =============================================================================

print("Loading model...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"   Loaded: {MODEL_PATH}")

# =============================================================================
# INTERACTIVE GENERATION
# =============================================================================

print("\n" + "=" * 50)
print("GPT Text Generator")
print("=" * 50)
print(f"   Temperature: {TEMPERATURE} | Tokens: {MAX_NEW_TOKENS} | Top-k: {TOP_K}")
print("   Type 'quit' to exit\n")

while True:
    prompt = input("Prompt: ").strip()
    
    if prompt.lower() == 'quit':
        break
    
    if not prompt:
        continue
    
    # Tokenize and generate
    tokens = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    output = model.generate(tokens, MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
    
    # Decode and display
    text = tokenizer.decode(output[0].tolist())
    print("\n" + "-" * 50)
    print(text)
    print("-" * 50 + "\n")
