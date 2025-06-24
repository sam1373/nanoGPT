#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py – nanoGPT+NSA multi‑GPU trainer with
          • RULER evaluation at several top‑k settings
          • short‑context coherency smoke‑test
          • token‑level trimming everywhere
          • miscellaneous robustness fixes

The file is intentionally complete; no lines are omitted.
"""

import os, time, math, pickle, json, glob, argparse
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
import wandb

from model import GPTConfig, GPT

# --------------------------------------------------------------------------
# 0.  configuration & CLI
# --------------------------------------------------------------------------
torch._dynamo.config.capture_scalar_outputs = True  # keep graphs whole

CFG = dict(
    # I/O --------------------------------------------------------------
    out_dir                   = 'out',
    eval_interval             = 1_000,
    log_interval              =   50,
    eval_iters                =  200,
    eval_only                 = False,
    always_save_checkpoint    = True,
    init_from                 = 'scratch',         # scratch | resume | dir/ckpt.pt
    dataset                   = 'openwebtext',
    # batching ---------------------------------------------------------
    batch_size                = 12,
    gradient_accumulation_steps = 1,
    train_seq_length          = 1024,              # original default
    # model spec -------------------------------------------------------
    n_layer                   =   12,
    n_head                    =   12,
    n_embd                    =  768,
    dropout                   = 0.0,
    bias                      = False,
    attention_type            = 'full',            # full | nsa
    nsa_block_size            = 64,
    nsa_topk                  = 16,
    nsa_num_kv_heads          = -1,
    nsa_window_size           = 512,
    nsa_local_blocks          = 4,
    # optimiser --------------------------------------------------------
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
    # system -----------------------------------------------------------
    backend                   = 'nccl',
    device                    = 'cuda',
    compile_model             = False,
    # --- RULER --------------------------------------------------------
    ruler_eval_enabled        = False,
    ruler_eval_dir            = 'data/ruler_tasks',
    ruler_eval_interval       = 100,
    ruler_samples_per_task    = 10,
    ruler_eval_max_new_tokens = 10,
    ruler_verbose             = False,
    ruler_temp                = 0.1,
    ruler_topk_list           = '1,10,50',
    # --- Coherency smoke‑test ----------------------------------------
    coherency_eval_enabled    = True,
    coherency_prompts         = 'Hello there!,The quick brown fox',
    coherency_max_new_tokens  = 32,
    # -----------------------------------------------------------------
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
# 1.  distributed bootstrap
# --------------------------------------------------------------------------
use_slurm = os.getenv('SLURM_JOB_ID') is not None
if use_slurm:
    os.environ['RANK']       = os.getenv('SLURM_PROCID')
    os.environ['WORLD_SIZE'] = os.getenv('SLURM_NTASKS')
    os.environ['LOCAL_RANK'] = os.getenv('SLURM_LOCALID')
    os.environ.setdefault('MASTER_ADDR', os.getenv('SLURM_NODELIST','').split(',')[0].split('(')[0])
    os.environ.setdefault('MASTER_PORT', '12910')

ddp = int(os.getenv('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
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
# 2.  misc setup
# --------------------------------------------------------------------------
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
DTYPE_MAP = {'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}
ptdtype   = DTYPE_MAP[model_dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
tokenizer = tiktoken.get_encoding('gpt2')

# ---- W&B ------------------------------------------------------------------
if master_process and enable_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name,
               config={k: v for k, v in CFG.items()
                       if isinstance(v, (int, float, str, bool))})

data_dir = os.path.join('data', dataset)

# --------------------------------------------------------------------------
# 3.  data loader
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
# 4.  model & optimiser
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

if ddp:
    model = DDP(model, device_ids=[int(device.split(':')[-1])], bucket_cap_mb=32)
    for fn in ('generate',):
        setattr(model, fn, getattr(model.module, fn))

eval_model = model.module if ddp else model

# --------------------------------------------------------------------------
# 5.  helpers
# --------------------------------------------------------------------------
def lr_at(it):
    if it < warmup_iters:
        return learning_rate * (it+1)/(warmup_iters+1)
    if it > lr_decay_iters:
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5*(1+math.cos(math.pi*r))*(learning_rate - min_lr)

def throughput(num_tokens, seconds):
    return (num_tokens / seconds) if seconds > 0 else 0.0

@torch.no_grad()
def estimate_loss():
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

# ---------------- RULER ----------------------------------------------------
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
def eval_ruler(tasks: Dict[str, List[dict]],
               topk_values: List[int],
               verbose=False):
    eval_model.eval()
    results: Dict[int, Dict[str, float]] = {k: {} for k in topk_values}

    for tname, samples in tasks.items():
        correct = {k: 0 for k in topk_values}
        total   = 0

        for s in samples:
            prompt   = s['input']
            targets  = [str(t) for t in s['outputs']]

            prompt_tok = tokenizer.encode(prompt)
            idx = torch.tensor(prompt_tok, device=device).unsqueeze(0)

            with ctx:
                out_cache = {}
                for k in topk_values:
                    out_cache[k] = eval_model.generate(
                        idx, ruler_eval_max_new_tokens,
                        temp=ruler_temp, top_k=k
                    )

            total += 1
            for k in topk_values:
                gen_tokens = out_cache[k][0, len(prompt_tok):].tolist()  # token‑level trim
                gen = tokenizer.decode(gen_tokens)
                if any(t in gen for t in targets):
                    correct[k] += 1

                if verbose and master_process:
                    short = prompt[-40:].replace('\n',' ')
                    print(f"[{tname} | k={k}] …{short} | gen:\"{gen[:60]}\" "
                          f"| ok:{correct[k]}/{total}")

        for k in topk_values:
            if total:
                results[k][tname] = 100 * correct[k] / total

    eval_model.train()
    return results

# --------------- Coherency smoke‑test -------------------------------------
@torch.no_grad()
def eval_coherency(prompts: List[str], max_new: int):
    """
    Generate short continuations for fixed prompts; useful to detect
    catastrophic decoding bugs in the training loop.
    """
    eval_model.eval()
    outs = []
    for p in prompts:
        prompt_tok = tokenizer.encode(p)
        idx = torch.tensor(prompt_tok, device=device).unsqueeze(0)
        with ctx:
            gen = eval_model.generate(idx, max_new,
                                      temp=0.8, top_k=50)

        new_tokens = gen[0, len(prompt_tok):].tolist()      # token‑level slice
        outs.append((p, tokenizer.decode(new_tokens)))
    eval_model.train()
    return outs

ruler_tasks = load_ruler_tasks() if ruler_eval_enabled else {}
ruler_topk_values = [int(x) for x in ruler_topk_list.split(',')]
coherency_prompts_list = [s.strip() for s in coherency_prompts.split(',')]

# --------------------------------------------------------------------------
# 6.  training loop
# --------------------------------------------------------------------------
t0 = time.time()                # last log-time
toks_since_log = 0
while True:
    lr = lr_at(iter_num)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # ---- evaluation ------------------------------------------------------
    if iter_num % eval_interval == 0:
        if ddp: torch.distributed.barrier()
        losses = estimate_loss()

        ruler_results = {}
        if ruler_eval_enabled and iter_num % ruler_eval_interval == 0:
            ruler_results = eval_ruler(ruler_tasks,
                                       ruler_topk_values,
                                       ruler_verbose and master_process)

        coherency_out = []
        if coherency_eval_enabled and master_process:
            coherency_out = eval_coherency(
                coherency_prompts_list,
                coherency_max_new_tokens
            )

        if master_process:
            print(f"iter {iter_num}: train {losses['train']:.4f}, "
                  f"val {losses['val']:.4f}")

            if coherency_out:
                for p, g in coherency_out:
                    print(f"[coherency] \"{p}\" → \"{g[:50]}…\"")

            if enable_wandb:
                log_dict = {
                    'iter':        iter_num,
                    'train/loss':  losses['train'],
                    'val/loss':    losses['val'],
                    'lr':          lr,
                }
                for k, res in ruler_results.items():
                    for tname, acc in res.items():
                        log_dict[f"ruler@k{k}/{tname}"] = acc
                wandb.log(log_dict)

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                raw = model.module if ddp else model
                torch.save({'model': raw.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num':   iter_num,
                            'best_val_loss': best_val_loss},
                           os.path.join(out_dir, f'ckpt_iter{iter_num}.pt'))
        if ddp: torch.distributed.barrier()

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

    toks_since_log += batch_size * train_seq_length * ddp_world_size
    if master_process and iter_num % log_interval == 0:
        dt = time.time() - t0
        print(f"iter {iter_num}: loss {loss.item()*gradient_accumulation_steps:.4f}, "
              f"{dt*1e3:.1f} ms, lr {lr:.2e}, "
              f"{throughput(toks_since_log, dt):.1f} tok/s")
        if enable_wandb:
            wandb.log({'iter': iter_num,
                       'train/loss_step': loss.item()*gradient_accumulation_steps,
                       'tok_per_sec': throughput(toks_since_log, dt)})
        t0 = time.time()
        toks_since_log = 0

    iter_num += 1
    if iter_num > max_iters:
        break

destroy_process_group() if ddp else None
if master_process and enable_wandb:
    wandb.finish()

