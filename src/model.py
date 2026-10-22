import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        if d % heads: raise ValueError("d_model must be divisible by n_heads")
        self.heads, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z): return z.view(b, t, self.heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(q), split(k), split(v),
             is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                nn.Linear(4*d, d), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model=128, n_heads=4,
                 n_layers=4, dropout=0.1, tie_embeddings=True):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings: self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.context_length: raise ValueError("Input exceeds context_length")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos)[None, :, :])
        logits = self.lm_head(self.final_ln(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss
