#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py  –  nanoGPT+NSA multi‑GPU trainer (patched)

Key fixes (see README.md for details):
  • Rank‑synchronised RULER evaluation (single barrier before & after).
  • DDP created with static_graph=True → removes rebuild_buckets() broadcast.
  • Longer distributed timeout (60 min) & clean NCCL env‑vars.
  • Optional InfiniBand plugin is disabled for single‑node jobs to avoid hangs.
  • Robust checkpoint handling + torch‑dynamo scalar capture as before.
"""

import os, time, math, pickle, json, glob, argparse, datetime
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
import wandb
from model import GPTConfig, GPT

# --------------------------------------------------------------------------
# 0.  environment hardening before *anything* touches NCCL -----------------
# --------------------------------------------------------------------------
os.environ.pop("NCCL_BLOCKING_WAIT", None)              # deprecated
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# Disable the buggy IBext RDMA plugin for on‑node training (kept overridable)
os.environ.setdefault("NCCL_NET", "^IBEXT")

# --------------------------------------------------------------------------
# 1.  configuration & CLI --------------------------------------------------
# --------------------------------------------------------------------------
torch._dynamo.config.capture_scalar_outputs = True   # keep graphs whole

CFG = dict(
    out_dir                   = 'out',
    eval_interval             = 1_000,
    log_interval              =   50,
    eval_iters                =  200,
    eval_only                 = False,
    always_save_checkpoint    = True,
    init_from                 = 'scratch',           # scratch | resume | dir/ckpt.pt
    dataset                   = 'openwebtext',
    batch_size                = 12,
    gradient_accumulation_steps = 1,
    train_seq_length          = 1024,
    n_layer                   =   12,
    n_head                    =   12,
    n_embd                    =  768,
    dropout                   = 0.0,
    bias                      = False,
    attention_type            = 'full',              # full | nsa
    nsa_block_size            = 64,
    nsa_topk                  = 16,
    nsa_num_kv_heads          = -1,
    nsa_window_size           = 512,
    nsa_local_blocks          = 4,
    learning_rate             = 6e-4,
    weight_decay              = 1e-1,
    beta1                     = 0.9,
    beta2                     = 0.95,
    max_iters                 = 600_000,
    grad_clip                 = 1.0,
    decay_lr                  = True,
    warmup_iters              = 2_000,
    lr_decay_iters            = 600_000,
    min_lr                    = 6e-5,
    backend                   = 'nccl',
    device                    = 'cuda',
    compile_model             = False,
    # --- RULER ------------------------------------------------------------
    ruler_eval_enabled        = False,
    ruler_eval_dir            = 'data/ruler_tasks',
    ruler_eval_interval       = 100,
    ruler_samples_per_task    = 10,
    ruler_eval_max_new_tokens = 10,
    ruler_verbose             = False,
    # ---------------------------------------------------------------------
    model_dtype               = ('bf16' if torch.cuda.is_available()
                                         and torch.cuda.is_bf16_supported()
                               else 'fp16'),
    enable_wandb              = False,
    wandb_run_name            = "run",
    wandb_project             = "nanogpt",
)

# ---- parse CLI -----------------------------------------------------------
parser = argparse.ArgumentParser()
for k, v in CFG.items():
    if isinstance(v, bool):
        parser.add_argument(f'--{k}', action='store_true' if not v else 'store_false', dest=k)
    elif k == 'model_dtype':
        parser.add_argument(f'--{k}', choices=['fp32', 'bf16', 'fp16'], default=v)
    else:
        parser.add_argument(f'--{k}', type=type(v), default=v)
CFG.update(vars(parser.parse_args()))
globals().update(CFG)

# --------------------------------------------------------------------------
# 2.  distributed bootstrap ------------------------------------------------
# --------------------------------------------------------------------------
use_slurm = os.getenv('SLURM_JOB_ID') is not None
if use_slurm:
    os.environ['RANK']       = os.getenv('SLURM_PROCID')
    os.environ['WORLD_SIZE'] = os.getenv('SLURM_NTASKS')
    os.environ['LOCAL_RANK'] = os.getenv('SLURM_LOCALID')
    os.environ.setdefault('MASTER_ADDR', os.getenv('SLURM_NODELIST','').split(',')[0].split('(')[0])
    os.environ.setdefault('MASTER_PORT', '12910')

# Longer watchdog timeout so big eval blocks don’t fire it
DDP_TIMEOUT = datetime.timedelta(minutes=60)

ddp = int(os.getenv('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend, timeout=DDP_TIMEOUT)
    ddp_rank        = int(os.environ['RANK'])
    ddp_local_rank  = int(os.environ['LOCAL_RANK'])
    ddp_world_size  = int(os.environ['WORLD_SIZE'])
    device          = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process  = (ddp_rank == 0)
    seed_offset     = ddp_rank
    if gradient_accumulation_steps % ddp_world_size:
        gradient_accumulation_steps //= ddp_world_size
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1
    device         = 'cuda'

tokens_per_iter = (gradient_accumulation_steps *
                   ddp_world_size * batch_size * train_seq_length)
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"tokens/iter: {tokens_per_iter:,}")

# --------------------------------------------------------------------------
# 3.  misc setup -----------------------------------------------------------
# --------------------------------------------------------------------------
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
DTYPE_MAP = {'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}
ptdtype   = DTYPE_MAP[model_dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# Tokeniser ---------------------------------------------------------------
try:
    tokenizer = tiktoken.get_encoding('gpt2')
except Exception:
    raise RuntimeError("tiktoken not found or GPT‑2 encoding unavailable – install tiktoken >=0.5.1")

if master_process and enable_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name)

data_dir = os.path.join('data', dataset)

# --------------------------------------------------------------------------
# 4.  data loader ----------------------------------------------------------
# --------------------------------------------------------------------------
train_memmap = np.memmap(os.path.join(data_dir,'train.bin'),dtype=np.uint16,mode='r')
val_memmap   = np.memmap(os.path.join(data_dir,'val.bin'  ),dtype=np.uint16,mode='r')

def get_batch(split):
    data = train_memmap if split == 'train' else val_memmap
    ix   = torch.randint(len(data) - train_seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+train_seq_length].copy()).long() for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+train_seq_length].copy()).long() for i in ix])
    if device.startswith('cuda'):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# --------------------------------------------------------------------------
# 5.  model & optimiser ----------------------------------------------------
# --------------------------------------------------------------------------
meta_vocab_size = None
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    meta_vocab_size = pickle.load(open(meta_path,'rb'))['vocab_size']

model_args = dict(
    n_layer          = n_layer,      n_head  = n_head,   n_embd = n_embd,
    train_seq_length = train_seq_length,
    bias             = bias,         vocab_size = meta_vocab_size or 50304,
    dropout          = dropout,      attention_type = attention_type,
    nsa_block_size   = nsa_block_size, nsa_topk = nsa_topk,
    nsa_num_kv_heads = nsa_num_kv_heads, nsa_window_size = nsa_window_size,
    nsa_local_blocks = nsa_local_blocks, model_dtype = model_dtype,
)

ckpt_files = sorted(glob.glob(os.path.join(out_dir,'ckpt_iter*.pt')), key=os.path.getmtime)
have_ckpt = bool(ckpt_files) and init_from == 'resume'

if have_ckpt:
    last_ckpt = ckpt_files[-1]
    if master_process: print(f"[loader] Resuming from {last_ckpt}")
    ckpt        = torch.load(last_ckpt, map_location='cpu')
    model_args.update(ckpt['model_args'])
    model       = GPT(GPTConfig(**model_args))
    model.load_state_dict(ckpt['model'])
    iter_num        = ckpt['iter_num']
    best_val_loss   = ckpt['best_val_loss']
else:
    if init_from == 'scratch':
        if master_process: print("[loader] Initialising from scratch")
        model = GPT(GPTConfig(**model_args))
    else:
        if master_process: print(f"[loader] Loading pretrained from {init_from}")
        model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    iter_num      = 0
    best_val_loss = 1e9

model.to(device)
if master_process:
    print(f"Model parameters: {model.get_num_params():,}")

scaler     = torch.amp.GradScaler(enabled=(model_dtype=='fp16'))
optimizer  = model.configure_optimizers(weight_decay, learning_rate,
                                        (beta1,beta2),'cuda')
if have_ckpt:
    optimizer.load_state_dict(ckpt['optimizer'])

if compile_model:
    model = torch.compile(model)

# Distributed wrapper ------------------------------------------------------
if ddp:
    model = DDP(
        model,
        device_ids=[int(device.split(':')[-1])],
        bucket_cap_mb=32,
        static_graph=True        # <- remove rebuild_buckets() broadcast
    )
    # expose generate() on the wrapper
    for fn in ('generate',):
        setattr(model, fn, getattr(model.module, fn))

# Object for *single-rank* inference (no DDP collectives)
eval_model = model.module if ddp else model

# --------------------------------------------------------------------------
# 6.  helpers --------------------------------------------------------------
# --------------------------------------------------------------------------

def lr_at(it):
    if it < warmup_iters:                 # linear warm-up
        return learning_rate * (it+1)/(warmup_iters+1)
    if it > lr_decay_iters:               # floor
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5*(1+math.cos(math.pi*r))*(learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    """Per-rank loss estimate -> world averaged."""
    eval_model.eval()
    out = {}
    for split in ('train','val'):
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            with ctx:
                _, l, _ = eval_model(xb, targets=yb)
            losses[i] = l
        if ddp:
            torch.distributed.all_reduce(losses)
            losses /= ddp_world_size
        out[split] = losses.mean().item()
    eval_model.train()
    return out

# ---------------- RULER ---------------------------------------------------

def load_ruler_tasks():
    tasks = {}
    if not os.path.isdir(ruler_eval_dir):
        if master_process:
            print(f"[RULER] directory not found: {ruler_eval_dir}")
        return tasks
    for fp in glob.glob(os.path.join(ruler_eval_dir,'*.jsonl')):
        with open(fp,'r',encoding='utf-8') as f:
            rows = [json.loads(l) for _,l in
                     zip(range(ruler_samples_per_task), f)]
        if rows:
            tasks[os.path.splitext(os.path.basename(fp))[0]] = rows
    if master_process:
        print(f"[RULER] loaded {len(tasks)} task files")
    return tasks

@torch.no_grad()
def eval_ruler(tasks, verbose=False):
    """Single-rank evaluation; use *eval_model*, no collectives."""
    eval_model.eval()
    results = {}
    for tname, samples in tasks.items():
        correct = total = 0
        pre_ms = dec_ms = pre_tok = dec_tok = 0.0
        for s in samples:
            prompt  = s['input']
            targets = list(map(str, s['outputs']))
            idx     = torch.tensor(tokenizer.encode(prompt),
                                   device=device).unsqueeze(0)
            pl      = idx.size(1)
            torch.cuda.synchronize(device); t0 = time.time()
            with ctx:
                _ = eval_model(idx)                # pre-fill
            torch.cuda.synchronize(device); t1 = time.time()
            with ctx:
                out = eval_model.generate(idx, ruler_eval_max_new_tokens,
                                          temp=0.1, top_k=1)
            torch.cuda.synchronize(device); t2 = time.time()
            pre_ms  += (t1-t0)*1e3;  dec_ms += (t2-t1)*1e3
            pre_tok += pre_ms/pl;    dec_tok += dec_ms/(out.size(1)-pl)
            gen = tokenizer.decode(out[0].tolist())[len(prompt):]
            correct += any(t in gen for t in targets);  total += 1
            if verbose and master_process:
                short = prompt[-40:].replace('\n',' ')
                print(f"[{tname}] …{short} | gen:\"{gen[:60]}\" | ok:{correct}/{total}")
        if total:
            results[tname] = (100*correct/total,
                              pre_ms/total, dec_ms/total,
                              pre_tok/total, dec_tok/total)
    eval_model.train()
    return results

ruler_tasks = load_ruler_tasks() if ruler_eval_enabled else {}

# --------------------------------------------------------------------------
# 7.  training loop --------------------------------------------------------
# --------------------------------------------------------------------------

t0 = time.time()
while True:
    # learning‑rate scheduler
    lr = lr_at(iter_num)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # ---- evaluation ------------------------------------------------------
    if iter_num % eval_interval == 0:
        if ddp:
            torch.distributed.barrier()       # sync before eval
        # losses on *all* ranks
        losses = estimate_loss()
        if master_process:
            print(f"iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                raw = model.module if ddp else model
                torch.save({'model': raw.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num':   iter_num,
                            'best_val_loss': best_val_loss},
                           os.path.join(out_dir, f'ckpt_iter{iter_num}.pt'))
            # ---- RULER on rank 0 only -----------------------------------
            if ruler_eval_enabled and iter_num % ruler_eval_interval == 0:
                ruler_metrics = eval_ruler(ruler_tasks, ruler_verbose)
                for tn,(acc,p,d,pt,dt) in ruler_metrics.items():
                    print(f"iter {iter_num} - {tn}: acc {acc:.2f}% | pre {p:.1f} ms | dec {d:.1f} ms")
        if ddp:
            torch.distributed.barrier()       # sync after eval

    if eval_only and iter_num == 0:
        break

    # ---- training step ---------------------------------------------------
    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps-1)
        xb, yb = get_batch('train')
        with ctx:
            _, loss, _ = model(xb, targets=yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer);  scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ---- logging ---------------------------------------------------------
    if master_process and iter_num % log_interval == 0:
        dt = time.time() - t0;  t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item()*gradient_accumulation_steps:.4f}, {dt*1e3:.1f} ms, lr {lr:.2e}")

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

