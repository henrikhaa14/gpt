# =============================================================================
# GPT Training Script (Colab Compatible)
# =============================================================================
# Train a small GPT model on custom text data.
# Works on both local machines and Google Colab with GPU acceleration.
#
# Colab Setup:
#   1. Runtime > Change runtime type > GPU
#   2. Upload your text file to inputs/ folder
#   3. Run this script
# =============================================================================

# --- Dependencies ---
# Uncomment to install in Colab:
# !pip install tiktoken

import tiktoken
import torch
import torch.nn as nn
from tqdm import tqdm  # Progress bars

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# File settings
INPUT_FILE = "inputs/tiny-shakespeare.txt"  # Path to training text
MODEL_SAVE_PATH = "models/gpt_model.pt"     # Where to save the model

# Model architecture
VOCAB_SIZE = 50257      # GPT-2 tokenizer vocab size (don't change)
EMBEDDING_DIM = 128     # Size of token embeddings
NUM_HEADS = 8           # Number of attention heads
NUM_LAYERS = 2          # Number of transformer blocks
MAX_SEQ_LEN = 512       # Maximum context window

# Training hyperparameters
BATCH_SIZE = 16         # Samples per batch (reduce if out of memory)
SEQ_LEN = 512           # Training sequence length
LEARNING_RATE = 1e-4    # Optimizer learning rate
NUM_EPOCHS = 3          # Number of training epochs

# =============================================================================
# DATA LOADING
# =============================================================================

print("📚 Loading data...")
with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Tokenize using GPT-2's tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)

print(f"   Characters: {len(text):,}")
print(f"   Tokens: {len(token_ids):,}")
print(f"   Steps/epoch: ~{len(token_ids) // SEQ_LEN:,}")

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    
    Each token attends to all previous tokens (and itself) to gather
    contextual information. The causal mask prevents looking at future tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Combined Q, K, V projection for efficiency
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence, Channels
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention: (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask: prevent attending to future tokens
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        
        # Reshape back and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Expands the representation, applies non-linearity, then contracts.
    This is where the model "thinks" about individual positions.
    """
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),    # Expand
            nn.GELU(),                        # Smooth activation
            nn.Linear(ff_dim, embed_dim),    # Contract
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block: LayerNorm -> Attention -> LayerNorm -> FFN
    
    Uses pre-norm architecture and residual connections for stable training.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Complete GPT language model.
    
    Architecture: Token Embed + Pos Embed -> N x TransformerBlock -> LN -> Head
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim=4 * embed_dim)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        
        # Token + positional embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


# =============================================================================
# TRAINING SETUP
# =============================================================================

print("\n🔧 Initializing...")

# Device selection (automatically uses GPU in Colab)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Initialize model
model = GPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(device)

# Print model size
num_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def create_batches(tokens: list, seq_len: int):
    """Generate training batches from token list."""
    for i in range(0, len(tokens) - seq_len, seq_len):
        batch = tokens[i:i + seq_len + 1]
        if len(batch) == seq_len + 1:
            yield batch


# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n🚀 Training started!\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_steps = 0
    
    # Create progress bar
    batches = list(create_batches(token_ids, SEQ_LEN))
    pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    for batch in pbar:
        # Prepare input (all but last) and target (all but first)
        batch = torch.tensor(batch, dtype=torch.long, device=device).unsqueeze(0)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        
        # Forward pass
        logits = model(inp)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track progress
        total_loss += loss.item()
        num_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Epoch summary
    avg_loss = total_loss / num_steps
    print(f"   Epoch {epoch + 1} complete | Avg loss: {avg_loss:.4f}\n")

# =============================================================================
# SAVE MODEL
# =============================================================================

print("💾 Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"   Saved to: {MODEL_SAVE_PATH}")
print("\n✅ Training complete!")
