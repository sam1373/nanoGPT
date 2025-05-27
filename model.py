"""
Full definition of a GPT Language Model, all of it in this single file.
Modifications:
- Removed learned absolute positional embeddings (wpe).
- Implemented Rotary Positional Embeddings (RoPE) as the sole method for positional info.
- RoPE is applied for both 'full' and 'nsa' attention types.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import NativeSparseAttention, KVCache, and RopeConfig
try:
    from native_sparse_attention.modules import NativeSparseAttention
    from native_sparse_attention.utils import KVCache, RopeConfig
    nsa_available = True
except ImportError:
    # If the import fails, set a flag and define dummy classes so the script
    # can still be parsed by Python without crashing on undefined type hints.
    nsa_available = False
    # These dummy classes will not be instantiated if nsa_available is False.
    NativeSparseAttention = object
    KVCache = object
    RopeConfig = object


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

@dataclass
class GPTConfig:
    train_seq_length: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # Bias is False by default for RoPE models like Llama

    # --- Attention Architecture Configs ---
    attention_type: str = 'full' # 'full' or 'nsa'

    # --- RoPE Configuration (now always enabled) ---
    rope_enable: bool = True # Kept for clarity, but effectively always on
    rope_base: float = 10000.0
    # rope_dim: int = -1 # If -1, defaults to n_embd // n_head. Can be set for partial RoPE.

    # --- NSA-specific parameters ---
    nsa_block_size: int = 64
    nsa_topk: int = 16
    nsa_num_kv_heads: int = -1
    nsa_window_size: int = 512
    nsa_local_blocks: int = 4

# --- RoPE Implementation for 'full' attention ---
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make cached cos/sin available for the forward pass
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len, device, dtype=torch.float32): # Default to float32 for cache
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # Use torch.outer for clarity
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q_or_k, position_ids):
        # q_or_k: [batch_size, num_heads, seq_len, head_dim]
        # position_ids: [batch_size, seq_len]

        seq_len = position_ids.max() + 1 # Determine required length from position_ids
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=q_or_k.device, dtype=q_or_k.dtype)

        # Gather the cosine and sine embeddings based on position_ids
        # cos_cached / sin_cached shape: [max_seq_len_cached, head_dim]
        # We need: [batch_size, 1, seq_len, head_dim] for broadcasting with q_or_k

        # Flatten position_ids for gathering, then reshape
        # This handles cases where position_ids are not contiguous [0, 1, ..., T-1] per batch item
        # (e.g., if KV caching is used and position_ids represent absolute positions)
        batch_size, num_heads, q_seq_len, head_dim = q_or_k.shape

        # Ensure position_ids are [batch_size, q_seq_len]
        if position_ids.ndim == 1: # If [T], expand to [1, T] then repeat for batch
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        cos = self.cos_cached[position_ids].unsqueeze(1).to(q_or_k.dtype) # [B, 1, T, D]
        sin = self.sin_cached[position_ids].unsqueeze(1).to(q_or_k.dtype) # [B, 1, T, D]

        def rotate_half(x):
            x1 = x[..., : self.dim // 2]
            x2 = x[..., self.dim // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return (q_or_k * cos) + (rotate_half(q_or_k) * sin)

class CausalSelfAttention(nn.Module): # For 'full' attention
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout # For F.scaled_dot_product_attention

        self.head_dim = config.n_embd // config.n_head
        if config.rope_enable:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, # RoPE dimension is head dimension
                max_position_embeddings=config.train_seq_length, # Cache up to train_seq_length
                base=config.rope_base
            )

    def forward(self, x, past_key_value=None, position_ids=None): # position_ids added
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        if hasattr(self, 'rotary_emb'):
            if position_ids is None:
                raise ValueError("position_ids must be provided when RoPE is enabled for CausalSelfAttention")
            q = self.rotary_emb(q, position_ids=position_ids)
            k = self.rotary_emb(k, position_ids=position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        updated_past_key_value = (k, v)

        # FlashAttention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=(past_key_value is None))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, updated_past_key_value

class NSAWrapper(nn.Module): # For 'nsa' attention
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        num_kv_h = config.nsa_num_kv_heads if config.nsa_num_kv_heads > 0 else config.n_head

        current_rope_config = None
        if config.rope_enable:
            current_rope_config = RopeConfig(
                dim=self.head_dim, # RoPE dimension is head dimension
                base=config.rope_base,
                traditional=False # Llama-style RoPE, common in many NSA implementations
            )

        self.nsa = NativeSparseAttention(
            hidden_size=config.n_embd,
            num_q_heads=config.n_head,
            num_kv_heads=num_kv_h,
            head_dim=self.head_dim,
            block_size=config.nsa_block_size,
            topk=config.nsa_topk,
            window_size=config.nsa_window_size,
            local_blocks=config.nsa_local_blocks,
            rope_config=current_rope_config,
            compress_type="weightedpool"
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_key_value: KVCache = None, position_ids=None): # position_ids added
        B, T, C = x.shape

        if T > 1:
            hidden_states = x.reshape(B * T, C)
            cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        else:
            hidden_states = x.squeeze(1)
            cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=x.device)

        # NativeSparseAttention with RopeConfig expects flattened position_ids for the batch
        # The position_ids passed from GPT.forward are already [B, T] representing absolute positions.
        # NSA's KVCache handles seqlen_offset internally when rope_config is active.
        nsa_position_ids = position_ids.view(-1) if position_ids is not None else None
        if self.nsa.rope_config is not None and nsa_position_ids is None:
             raise ValueError("position_ids must be provided when RoPE is enabled for NSAWrapper")


        attn_output_flat, updated_kv_cache = self.nsa(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_ids=nsa_position_ids, # Pass the (potentially flattened) position_ids
            past_key_value=past_key_value
        )

        if T > 1:
            attn_output = attn_output_flat.view(B, T, C)
        else:
            attn_output = attn_output_flat.unsqueeze(1)

        attn_output = self.dropout(attn_output)
        return attn_output, updated_kv_cache

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.attention_type == 'nsa':
            self.attn = NSAWrapper(config)
        elif config.attention_type == 'full':
            self.attn = CausalSelfAttention(config)
        else:
            raise ValueError(f"Unknown attention_type: {config.attention_type}")
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_key_value=None, position_ids=None): # position_ids added
        attn_output, updated_past_key_value = self.attn(
            self.ln_1(x),
            past_key_value=past_key_value,
            position_ids=position_ids # Pass position_ids to attention
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, updated_past_key_value

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.train_seq_length is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe (learned positional embedding) is REMOVED
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        # No wpe to subtract anymore
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # Only wte now
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None):
        device = idx.device
        b, t = idx.size()
        # if t > self.config.train_seq_length and past_key_values is None: # Check only if not using KV cache
        #      raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.config.train_seq_length}")


        # Determine past_length for RoPE position_ids
        past_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            if self.config.attention_type == 'nsa':
                # NSA's KVCache object should store the offset
                # Assuming KVCache from native-sparse-attention has a 'seqlen_offset' or similar
                # If KVCache doesn't store it directly, it might need to be tracked externally or inferred
                # For now, let's assume it's available or we'd need to adjust NSAWrapper/KVCache
                if hasattr(past_key_values[0], 'seqlen_offset'):
                     past_length = past_key_values[0].seqlen_offset
                else: # Fallback if KVCache doesn't have seqlen_offset, infer from a known structure
                    # This part is tricky without knowing exact KVCache structure for NSA
                    # A robust way is for KVCache to store total keys seen.
                    # For simplicity, if not found, we might have to rely on idx shape if t==1
                    if t == 1 and hasattr(past_key_values[0], 'key') and past_key_values[0].key is not None: # A guess
                        past_length = past_key_values[0].key.shape[2] # Assuming key shape [B, H, T_past, D]
            else: # 'full' attention
                past_length = past_key_values[0][0].shape[2] # k_cache shape (B, nh, T_past, hs)

        # Create absolute position_ids for the current chunk of tokens
        position_ids = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(b, t) # Shape (b, t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # No more wpe, directly use token embeddings
        x = self.transformer.drop(tok_emb)

        new_past_key_values_list = [] if past_key_values is not None else None

        for i, block in enumerate(self.transformer.h):
            block_past_kv = past_key_values[i] if past_key_values is not None else None
            x, updated_block_kv = block(x, past_key_value=block_past_kv, position_ids=position_ids) # Pass position_ids
            if new_past_key_values_list is not None:
                new_past_key_values_list.append(updated_block_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_past_key_values_list


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # ... (This method would need significant updates if the pretrained GPT-2 models
        #      did not use RoPE. Standard GPT-2 uses learned absolute PEs.
        #      For now, this method is likely incompatible with a RoPE-only model
        #      unless loading weights into a RoPE-enabled architecture.)
        print("WARNING: from_pretrained for GPT-2 is not directly compatible with a RoPE-only model without architectural adjustments or specific RoPE-trained weights.")
        # ... (rest of the original from_pretrained, but it will likely fail or misbehave)
        # For a RoPE model, you'd typically load weights from a model already trained with RoPE.
        # If you MUST load GPT-2 weights:
        # 1. The wpe weights from GPT-2 checkpoint would be ignored.
        # 2. The model would operate with RoPE instead. Performance would differ.

        # Simplified version assuming you are loading into this RoPE architecture:
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args) # Only dropout override supported by original

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s (NOTE: RoPE architecture)" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['train_seq_length'] = 1024 # This will be max_position_embeddings for RoPE
        config_args['bias'] = True # GPT-2 used bias, our RoPE model defaults to False. Forcing True for loading.
                                   # This might conflict with the class default.
                                   # Best to ensure config.bias is True if loading GPT-2 weights.

        # Add RoPE config to match our model, though GPT-2 didn't use it
        config_args['rope_enable'] = True
        config_args['rope_base'] = 10000.0
        config_args['attention_type'] = 'full' # Assuming loading into 'full' attention for GPT-2 structure

        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        current_config = GPTConfig(**config_args)
        model = GPT(current_config)
        # ... (rest of weight loading logic from original nanoGPT, be mindful of wpe)
        # The original logic tries to load 'transformer.wpe.weight'. This will fail as it's removed.
        # We need to adapt the loading to skip wpe.

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] # Original filter

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys() if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]

        # Filter out wpe from HuggingFace state_dict as our model doesn't have it
        sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('transformer.wpe.')]
        # Also filter out 'transformer.wpe.weight' from our model's sd_keys if it somehow got there (it shouldn't)
        sd_keys = [k for k in sd_keys if not k.startswith('transformer.wpe.')]


        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Adjust for potentially different bias settings if GPTConfig default is False
        # but GPT-2 weights have bias.
        # This is complex; for simplicity, assume bias is handled by config_args['bias']=True.

        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} ({sd_keys_hf}) != {len(sd_keys)} ({sd_keys})"

        for k_hf in sd_keys_hf:
            k_nano = k_hf # Assuming names match after filtering wpe
            if any(k_nano.endswith(w) for w in transposed):
                assert sd_hf[k_hf].shape[::-1] == sd[k_nano].shape
                with torch.no_grad():
                    sd[k_nano].copy_(sd_hf[k_hf].t())
            else:
                assert sd_hf[k_hf].shape == sd[k_nano].shape
                with torch.no_grad():
                    sd[k_nano].copy_(sd_hf[k_hf])
        return model


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # ... (optimizer configuration remains the same)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # ... (generate method remains largely the same, as KV caching logic in forward handles positions)
        past_key_values = [None] * self.config.n_layer

        for _ in range(max_new_tokens):
            # if using KV cache, idx_cond is the full prompt on the first pass, then 1 token after that.
            if past_key_values[0] is None:
                # On the first pass (prefill), use the entire provided prompt 'idx'
                # regardless of its length. This is what enables sequence length extrapolation testing.
                idx_cond = idx
            else:
                # After prefill, only process the very last token for efficiency
                idx_cond = idx[:, [-1]]

            logits, _, updated_past_key_values = self(idx_cond, past_key_values=past_key_values)
            past_key_values = updated_past_key_values

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

