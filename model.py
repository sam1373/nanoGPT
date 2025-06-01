"""
model.py – GPT-style LM with optional Native-Sparse-Attention (NSA)
------------------------------------------------------------------
*  full attention  → Flash-style causal dot-product
*  nsa attention   → Native-Sparse-Attention Triton kernels
"""

from __future__ import annotations
import math, inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# native-sparse-attention import guard
# ---------------------------------------------------------------------------
try:
    from native_sparse_attention.module import NativeSparseAttention, RopeConfig, NSACache
    _nsa_ok = True
except ImportError:
    _nsa_ok = False
    class _Missing:                                   # type: ignore
        def __getattr__(self, _):                     # noqa: D401
            raise RuntimeError("native-sparse-attention not installed.")
    NativeSparseAttention = RopeConfig = NSACache = _Missing  # type: ignore

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

class LayerNorm(nn.Module):
    def __init__(self, n: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias   = nn.Parameter(torch.zeros(n)) if bias else None
    def forward(self, x):                              # type: ignore[override]
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    train_seq_length: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    attention_type: str = "full"           # "full" | "nsa"
    rope_base: float = 10000.0

    # NSA hyper-params
    nsa_block_size: int = 64
    nsa_topk: int = 16
    nsa_num_kv_heads: int = -1
    nsa_window_size: int = 512
    nsa_local_blocks: int = 4
    nsa_kernel_size: int = 32
    nsa_kernel_stride: int = 16
    nsa_compress_type: str = "weightedpool"

    model_dtype: str = "bf16"              # "bf16" | "fp16" | "fp32"

# ---------------------------------------------------------------------------
# rotary embedding
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)
        self._build(max_pos, torch.float32)

    def _build(self, seq: int, dtype: torch.dtype):
        t = torch.arange(seq, device=self.inv.device, dtype=self.inv.dtype)
        emb = torch.outer(t, self.inv).repeat(1, 2)
        self.register_buffer("cos", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin", emb.sin().to(dtype), persistent=False)
        self.max_seq = seq

    def forward(self, x, pos):
        if int(pos.max()) + 1 > self.max_seq:
            self._build(int(pos.max()) + 1, x.dtype)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        cos = self.cos[pos].unsqueeze(1).to(x.dtype)
        sin = self.sin[pos].unsqueeze(1).to(x.dtype)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rot = torch.cat([-x2, x1], -1)
        return x * cos + rot * sin

# ---------------------------------------------------------------------------
# full flash-style attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.h = cfg.n_head
        self.d = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.rot  = RotaryEmbedding(self.d, cfg.train_seq_length, cfg.rope_base)

    def forward(self, x, past, pos):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)
        q, k = self.rot(q, pos), self.rot(k, pos)

        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        new = (k, v)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=(self.drop.p if self.training else 0.0),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y)), new

# ---------------------------------------------------------------------------
# NSA wrapper
# ---------------------------------------------------------------------------
class NSAAttention(nn.Module):
    """
    NSA attention wrapper:
      - Training / validation: var-len kernel (self.nsa(...))
      - Inference:
          * If `past` is None and T>1: do var-len kernel to get output,
            then build a new cache by single-token steps in a loop.
          * If `past` is not None (decoding step): T==1 call nsa.inference(...).
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if not _nsa_ok:
            raise RuntimeError("native-sparse-attention not installed")

        self.comp_dtype = (
            torch.bfloat16 if cfg.model_dtype == "bf16" else
            torch.float16  if cfg.model_dtype == "fp16" else
            torch.float32
        )

        head_dim = cfg.n_embd // cfg.n_head
        num_kv = cfg.nsa_num_kv_heads if cfg.nsa_num_kv_heads > 0 else cfg.n_head

        rope_cfg = RopeConfig(head_dim=head_dim, rope_theta=cfg.rope_base)

        self.nsa = NativeSparseAttention(
            hidden_size   = cfg.n_embd,
            num_q_heads   = cfg.n_head,
            num_kv_heads  = num_kv,
            head_dim      = head_dim,
            compress_type = cfg.nsa_compress_type,
            kernel_size   = cfg.nsa_kernel_size,
            kernel_stride = cfg.nsa_kernel_stride,
            block_size    = cfg.nsa_block_size,
            topk          = cfg.nsa_topk,
            init_blocks   = 1,
            local_blocks  = cfg.nsa_local_blocks,
            window_size   = cfg.nsa_window_size,
            rope_config   = rope_cfg,
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # ensure any internal buffers are in the correct dtype
        for name in ("compress_key", "compress_value", "intra_block_pe"):
            buf = getattr(self.nsa, name, None)
            if buf is not None and buf.dtype != self.comp_dtype:
                buf.data = buf.data.to(self.comp_dtype)

    @staticmethod
    def _flat(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flatten batch for NSA var-len kernel:
          x -> shape [batch_size, seq_len, hidden] => [B*T, hidden]
          cu -> cumsum of seq_lens across batch
        """
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        cu = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        return flat, cu

    def _new_cache(self, batch: int, device: torch.device) -> NSACache:
        return NSACache(
            max_batch_size=batch,
            max_length=100_000,
            num_kv_heads=self.nsa.num_kv_heads,
            head_dim=self.nsa.head_dim,
            kernel_size=self.nsa.kernel_size,
            kernel_stride=self.nsa.kernel_stride,
            window_size=self.nsa.window_size,
            dtype=self.comp_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, past: Optional[NSACache], pos: torch.Tensor):
        """
        x: [B, T, C]
        past: NSA cache or None
        pos: [B, T]
        """
        src_dtype = x.dtype
        if src_dtype != self.comp_dtype:
            x = x.to(self.comp_dtype)

        B, T, C = x.shape
        flat, cu = self._flat(x)

        # Training or validation: just call var-len kernel
        if self.training:
            out = self.nsa(flat, cu)
            out = out.view(B, T, C).to(src_dtype)
            return self.dropout(out), None

        # Inference
        if past is None:
            # "prefill" mode with T > 1
            # single var-len call for the entire chunk
            out = self.nsa(flat, cu)
            out = out.view(B, T, C).to(src_dtype)

            # build a fresh cache ONCE for all tokens
            new_cache = self._new_cache(B, x.device)

            # do the "inference" path for the entire T in a single pass
            # "step=0" for the entire chunk
            # That means the kernel inside .inference() must handle T tokens
            # as a single chunk. Just as your reference code "prefills" in one shot.
            self.nsa.inference(flat, cu, 0, new_cache)

            out = self.dropout(out)
            return out, new_cache
        else:
            # incremental decode: T must be 1
            if T != 1:
                raise ValueError("NSA decode path expects a single token (T==1)")
            # step = pos[0,0], or sometimes you pass step in separately
            step = int(pos[0, 0])

            out = self.nsa.inference(flat, cu, step, past)
            out = out.view(B, 1, C).to(src_dtype)
            return self.dropout(out), past


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):  # type: ignore[override]
        return self.drop(self.proj(self.act(self.fc(x))))

# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        if cfg.attention_type == "nsa":
            if not _nsa_ok:
                raise RuntimeError("native-sparse-attention not installed")
            self.attn = NSAAttention(cfg)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, past, pos):
        y, new = self.attn(self.ln1(x), past, pos)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, new

# ---------------------------------------------------------------------------
# GPT wrapper
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.mdtype = DTYPE_MAP[cfg.model_dtype]

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Per-layer scaled init for attention proj
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

        self.to(self.mdtype)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, past_kv: Optional[List] = None):
        """
        idx: [B, T] tokens
        targets: optional [B, T] for loss
        past_kv: either None or list of length n_layer with the KV cache
        """
        B, T = idx.shape
        past_kv = past_kv or [None] * self.cfg.n_layer

        # figure out the 'past_len' from the first layer's cache
        if past_kv[0] is None:
            past_len = 0
        else:
            if self.cfg.attention_type == "nsa":
                # NSACache doesn't store shape directly but does track cur_len
                past_len = int(getattr(past_kv[0], "cur_len", 0))
            else:
                # standard tuple (k, v) approach
                past_len = past_kv[0][0].shape[2]

        pos = torch.arange(past_len, past_len + T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.drop(self.wte(idx).to(self.mdtype))

        new_kv = []
        for blk, p in zip(self.h, past_kv):
            x, p = blk(x, p, pos)
            new_kv.append(p)

        x = self.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            return logits, loss, None

        # for inference, return only the last token's logits
        logits = self.lm_head(x[:, -1:])
        return logits, None, new_kv

    @torch.no_grad()
    def generate(self, idx, max_new, temp=1.0, top_k=None):
        """
        Autoregressive sampling:
          - If there's a large initial prompt, pass it in as 'idx' once.
          - Then subsequent tokens are single steps, updating the past_kv.
        """
        # Start with no past_kv
        past = [None] * self.cfg.n_layer

        # We run for `max_new` steps
        for _ in range(max_new):
            # For the very first call with past[0] is None, `forward` will do
            # either full NSA var-len if T>1, or single step if T=1.
            logits, _, past = self(
                idx if past[0] is None else idx[:, -1:], past_kv=past
            )
            logits = logits[:, -1] / temp
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            next_tok = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    def configure_optimizers(self, wd, lr, betas, device_type):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused_ok = fused_ok and (device_type == "cuda")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused_ok)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

