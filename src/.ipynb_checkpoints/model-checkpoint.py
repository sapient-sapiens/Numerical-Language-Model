import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 

class ModelConfig: # Bias set to false by default, not part of config. 
    d_model = 256  
    vocab_size = 20 
    n_layer = 6 
    n_head = 8 
    dropout = 0.05
    max_len = 40 

class MLP(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.c_fc = nn.Linear(config.d_model, 4*config.d_model) 
        self.gelu = nn.GELU() 
        self.c_proj = nn.Linear(4*config.d_model, config.d_model) 
        self.dropout = nn.Dropout(config.dropout) 
    def forward(self, x):
        x = self.c_fc(x) 
        x = self.gelu(x) 
        x = self.c_proj(x) 
        x = self.dropout(x) 
        return x 

class CausalSelfAttention(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.c_attn = nn.Linear(config.d_model, config.d_model*3, bias = False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias = False) 
        self.attn_dropout = nn.Dropout(config.dropout) 
        self.resid_dropout = nn.Dropout(config.dropout) 
        self.n_head = config.n_head 
        self.d_model = config.d_model 
        self.dropout = config.dropout
        # RoPE setup
        self.head_dim = self.d_model // self.n_head
        assert self.head_dim % 2 == 0, "RoPE requires head_dim to be even"
        self.max_seq_len = getattr(config, 'max_len', 2048)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat((-x2, x1), dim=-1)

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (t, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (t, head_dim)
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,t,head_dim)
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,t,head_dim)
        return cos, sin

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (b, n_head, t, head_dim)
        b, h, t, d = q.shape
        cos, sin = self._rope_cache(t, q.device, q.dtype)
        # reshape cos/sin to (1,1,t,d) for broadcasting
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
    
    def _causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Shapes: (b, h, t, d)
        d = q.size(-1)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        t = q.size(-2)
        causal_mask = torch.triu(torch.ones((t, t), device=q.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        if self.training and self.dropout > 0:
            att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        return y
    def forward(self, x): 
        b, t, c = x.size() 
        # project to qkv and split
        q, k, v = self.c_attn(x).split(c, dim=2)
        # reshape to (b, n_head, t, head_dim)
        k = k.view(b, t, self.n_head, c//self.n_head).transpose(1, 2) 
        q = q.view(b, t, self.n_head, c//self.n_head).transpose(1, 2) 
        v = v.view(b, t, self.n_head, c//self.n_head).transpose(1, 2) 
        # apply RoPE to q and k
        q, k = self._apply_rope(q, k)
        y = self._causal_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(b, t, c) 
        y = self.resid_dropout(self.c_proj(y)) 
        return y 



class Block(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.ln1 = nn.LayerNorm(config.d_model) 
        self.ln2 = nn.LayerNorm(config.d_model) 
        self.attn = CausalSelfAttention(config) 
        self.mlp = MLP(config) 
    def forward(self, x): 
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x)) 
        return x 


class NLM(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.d_model = config.d_model 
        self.transformer = nn.ModuleDict(dict(
            te = nn.Embedding(config.vocab_size, config.d_model), 
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False) 
        self.lm_head.weight = self.transformer.te.weight 
        self.apply(self._init_weights) 
        for pn, p in self.named_parameters(): 
            if pn.endswith('c_proj.weight'): 
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6, ))
    def get_num_params(self, non_embedding=True): 
        n_params = sum(p.numel() for p in self.parameters()) 
        return n_params 
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02) 
        elif isinstance(module, nn.Embedding): 
            nn.init.normal_(module.weight, mean=0.0, std = 0.02) 
    def forward(self, idx, targets): 
        b, t = idx.size() 
        tok_emb = self.transformer.te(idx) 
        x = self.transformer.drop(tok_emb) 
        for block in self.transformer.h: 
            x = block(x) 
        x = self.transformer.ln_f(x) 
        if targets is not None: 
            logits = self.lm_head(x) 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: 
            logits = self.lm_head(x[:, [-1], :])
            loss = None 
        return logits, loss 

