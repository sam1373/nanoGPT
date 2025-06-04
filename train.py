import os
import time, math, pickle, json, glob, argparse
from contextlib import nullcontext
import numpy as np
import torch
import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
import wandb
from model import GPTConfig, GPT

# CRITICAL: Check that NCCL environment variables are set
print("NCCL_TIMEOUT =", os.environ.get("NCCL_TIMEOUT"))
print("NCCL_DEBUG =", os.environ.get("NCCL_DEBUG"))
print("NCCL_BLOCKING_WAIT =", os.environ.get("NCCL_BLOCKING_WAIT"))

torch._dynamo.config.capture_scalar_outputs = True

CFG = dict(
    out_dir='out',
    eval_interval=1000,
    log_interval=50,
    eval_iters=200,
    eval_only=False,
    always_save_checkpoint=True,
    init_from='scratch',
    dataset='openwebtext',
    batch_size=12,
    gradient_accumulation_steps=1,
    train_seq_length=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=False,
    attention_type='full',
    nsa_block_size=64,
    nsa_topk=16,
    nsa_num_kv_heads=-1,
    nsa_window_size=512,
    nsa_local_blocks=4,
    learning_rate=6e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    max_iters=600000,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=2000,
    lr_decay_iters=600000,
    min_lr=6e-5,
    backend='nccl',
    device='cuda',
    compile_model=False,
    ruler_eval_enabled=False,
    ruler_eval_dir='data/ruler_tasks',
    ruler_eval_interval=100,
    ruler_samples_per_task=10,
    ruler_eval_max_new_tokens=10,
    ruler_verbose=False,
    model_dtype=('bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16'),
    enable_wandb=False,
    wandb_run_name="my_default_run",
    wandb_project="my_default_project"
)

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
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    if gradient_accumulation_steps % ddp_world_size:
        gradient_accumulation_steps //= ddp_world_size
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    print(f"Rank {ddp_rank}: Final gradient_accumulation_steps = {gradient_accumulation_steps}")
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = 'cuda'
    print(f"Single GPU: gradient_accumulation_steps = {gradient_accumulation_steps}")

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * train_seq_length
if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
DTYPE_MAP = {
    'fp32': torch.float32,
    'float32': torch.float32,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'fp16': torch.float16,
    'float16': torch.float16
}
ptdtype = DTYPE_MAP[model_dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if master_process and enable_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name)

data_dir = os.path.join('data', dataset)

def get_batch(split):
    fn = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - train_seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + train_seq_length].copy()).long() for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + train_seq_length].copy()).long() for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = pickle.load(open(meta_path, 'rb'))['vocab_size'] if os.path.exists(meta_path) else None
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    train_seq_length=train_seq_length,
    bias=bias,
    vocab_size=meta_vocab_size or 50304,
    dropout=dropout,
    attention_type=attention_type,
    nsa_block_size=nsa_block_size,
    nsa_topk=nsa_topk,
    nsa_num_kv_heads=nsa_num_kv_heads,
    nsa_window_size=nsa_window_size,
    nsa_local_blocks=nsa_local_blocks,
    model_dtype=model_dtype
)

ckpt_files = sorted(glob.glob(os.path.join(out_dir, 'ckpt_iter*.pt')), key=os.path.getmtime)
if init_from == 'resume' and ckpt_files:
    last_ckpt = ckpt_files[-1]
    ckpt = torch.load(last_ckpt, map_location='cpu')
    model_args.update(ckpt['model_args'])
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(ckpt['model'])
    iter_num = ckpt['iter_num']
    best_val_loss = ckpt['best_val_loss']
else:
    if init_from == 'scratch':
        model = GPT(GPTConfig(**model_args))
    else:
        model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    iter_num = 0
    best_val_loss = 1e9

# Move model to device before DDP
model.to(device)
checkpoint_model_args = model_args.copy()

scaler = torch.amp.GradScaler('cuda', enabled=(model_dtype == 'fp16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and ckpt_files:
    optimizer.load_state_dict(ckpt['optimizer'])

# Compile model if requested (best practice: after .to(device), before DDP)
if compile_model:
    model = torch.compile(model)

# Wrap in DDP with small bucket size
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        bucket_cap_mb=4,  # Keep buckets small to avoid stalls
        broadcast_buffers=False
    )
    # Expose helper methods from the wrapped module
    for fn in ("generate",):
        if hasattr(model.module, fn):
            setattr(model, fn, getattr(model.module, fn))

tokenizer = tiktoken.get_encoding('gpt2')

def lr_at(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            with ctx:
                _, loss, _ = model(xb, targets=yb)
            losses[i] = loss.item()
        if ddp:
            torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM)
            losses /= ddp_world_size
        out[split] = losses.mean().item()
    model.train()
    return out

def save_ckpt():
    raw = model.module if ddp else model
    path = os.path.join(out_dir, f'ckpt_iter{iter_num}.pt')
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": checkpoint_model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss
    }, path)
    latest = os.path.join(out_dir, 'ckpt_latest.pt')
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        pass

def load_ruler_tasks():
    tasks = {}
    if not os.path.isdir(ruler_eval_dir):
        if master_process:
            print(f"[RULER] directory not found: {ruler_eval_dir}")
        return tasks
    files = glob.glob(os.path.join(ruler_eval_dir, '*.jsonl'))
    if master_process:
        print(f"[RULER] found {len(files)} task files in {ruler_eval_dir}")
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            rows = [json.loads(l) for _, l in zip(range(ruler_samples_per_task), f)]
        if rows:
            tasks[os.path.splitext(os.path.basename(fp))[0]] = rows
            if master_process:
                print(f"[RULER]   {os.path.basename(fp):<20}: {len(rows)} samples loaded")
    return tasks

@torch.no_grad()
def eval_ruler(tasks, verbose=False):
    model.eval()
    results = {}
    for tname, samples in tasks.items():
        correct = 0
        total = 0
        prefill_times = []
        decode_times = []
        prefill_per_token_times = []
        decode_per_token_times = []
        for i, s in enumerate(samples):
            prompt = s['input']
            targets = [str(x) for x in s['outputs']]
            idx = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
            prompt_len = idx.size(1)
            if device_type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.time()
            with ctx:
                _ = model(idx)
            if device_type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.time()
            with ctx:
                out = model.generate(idx, ruler_eval_max_new_tokens, temp=0.1, top_k=1)
            if device_type == 'cuda':
                torch.cuda.synchronize(device)
            t2 = time.time()
            pre_ms = (t1 - t0) * 1000
            dec_ms = (t2 - t1) * 1000
            prefill_times.append(pre_ms)
            decode_times.append(dec_ms)
            gen_len = out.size(1) - prompt_len
            prefill_per_token_times.append(pre_ms / max(prompt_len, 1))
            decode_per_token_times.append(dec_ms / max(gen_len, 1))
            gen = tokenizer.decode(out[0].tolist())[len(prompt):]
            ok = any(t in gen for t in targets)
            if verbose:
                short_prompt = prompt[-40:].replace('\n', ' ')
                print(f'[{tname} #{i:02d}] …{short_prompt} | pre:{pre_ms:6.1f} ms dec:{dec_ms:6.1f} ms | gen:"{gen[:60]}" | exp:{targets} | {"✔" if ok else "✘"}')
            correct += ok
            total += 1
        acc = 100 * correct / total if total else 0.0
        avg_pre = float(np.mean(prefill_times)) if prefill_times else 0.0
        avg_dec = float(np.mean(decode_times)) if decode_times else 0.0
        avg_pre_token = float(np.mean(prefill_per_token_times)) if prefill_per_token_times else 0.0
        avg_dec_token = float(np.mean(decode_per_token_times)) if decode_per_token_times else 0.0
        results[tname] = (acc, avg_pre, avg_dec, avg_pre_token, avg_dec_token)
    model.train()
    return results

ruler_tasks = load_ruler_tasks() if ruler_eval_enabled else {}
print(f"tokens/iter: {tokens_per_iter:,}")

# Optional single barrier for debugging - remove once stable
if ddp:
    torch.distributed.barrier()

t0 = time.time()
while True:
    for g in optimizer.param_groups:
        g['lr'] = lr_at(iter_num)

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        if enable_wandb:
            wandb.log({"iter": iter_num, "train_loss": losses['train'], "val_loss": losses['val']}, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            save_ckpt()
        if ruler_eval_enabled and iter_num % ruler_eval_interval == 0:
            ruler_metrics = eval_ruler(ruler_tasks, ruler_verbose)
            for tname, (acc, pre_ms, dec_ms, pre_token, dec_token) in ruler_metrics.items():
                print(f"iter {iter_num} - {tname}: RULER accuracy {acc:.2f}% | prefill {pre_ms:.1f} ms | decode {dec_ms:.1f} ms | prefill/token {pre_token:.2f} ms | decode/token {dec_token:.2f} ms")
                if enable_wandb:
                    wandb.log({
                        f"ruler_accuracy_{tname}": acc,
                        f"ruler_prefill_ms_{tname}": pre_ms,
                        f"ruler_decode_ms_{tname}": dec_ms,
                        f"ruler_prefill_ms_per_token_{tname}": pre_token,
                        f"ruler_decode_ms_per_token_{tname}": dec_token
                    }, step=iter_num)

    if iter_num == 0 and eval_only:
        break

    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
        xb, yb = get_batch('train')
        with ctx:
            _, loss, _ = model(xb, targets=yb)
            loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if master_process and iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, {dt * 1000:.1f} ms")
        if enable_wandb:
            wandb.log({"iter": iter_num, "loss": loss.item() * gradient_accumulation_steps, "time_ms": dt * 1000}, step=iter_num)

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
