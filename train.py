"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP using Slurm on multiple nodes:
$ sbatch your_slurm_script.sbatch
(The script will auto-detect the Slurm environment)
"""

import os
import time
import math
import pickle
import json # Added for RULER sample loading
from contextlib import nullcontext
import glob # Added for RULER directory scanning
import argparse # Added for safer configuration

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken # Added for RULER evaluation

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config values
# These can be overridden by command-line arguments
# I/O
out_dir_default = 'out'
eval_interval_default = 2000
log_interval_default = 1
eval_iters_default = 200
eval_only_default = False # if True, script exits right after the first eval
always_save_checkpoint_default = True # if True, always save a checkpoint after each eval
init_from_default = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log_default = False # disabled by default
wandb_project_default = 'owt'
wandb_run_name_default = 'gpt2' # 'run' + str(time.time())
# data
dataset_default = 'openwebtext'
gradient_accumulation_steps_default = 1 # used to simulate larger batch sizes
batch_size_default = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
train_seq_length_default = 1024
# model
n_layer_default = 12
n_head_default = 12
n_embd_default = 768
dropout_default = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias_default = False # do we use bias inside LayerNorm and Linear layers?
attention_type_default = 'full' # 'full' or 'nsa'
nsa_block_size_default = 64
nsa_topk_default = 16
nsa_num_kv_heads_default = -1
nsa_window_size_default = 512
nsa_local_blocks_default = 4
rope_enable_default = False # Placeholder for RoPE implementation

# adamw optimizer
learning_rate_default = 6e-4 # max learning rate
max_iters_default = 600000 # total number of training iterations
weight_decay_default = 1e-1
beta1_default = 0.9
beta2_default = 0.95
grad_clip_default = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr_default = True # whether to decay the learning rate
warmup_iters_default = 2000 # how many steps to warm up for
lr_decay_iters_default = 600000 # should be ~= max_iters per Chinchilla
min_lr_default = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend_default = 'nccl' # 'nccl', 'gloo', etc.
# system
device_default = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype_default = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile_default = True # use PyTorch 2.0 to compile the model to be faster

# --- RULER Benchmark Evaluation Config ---
ruler_eval_enabled_default = False # set to True to run RULER eval during validation
ruler_eval_dir_default = 'data/ruler_tasks' # path to folder with RULER JSONL task files
ruler_eval_interval_default = 100 # how often to run RULER evaluation (iterations)
ruler_samples_per_task_default = 10 # number of samples to evaluate from each RULER task file
ruler_eval_max_new_tokens_default = 10 # max tokens to generate for RULER prompts
# -----------------------------------------------------------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a GPT model.")
# Add arguments based on the default config values
# Note: Using a loop or a more structured way to define these might be cleaner for very many args
parser.add_argument('--out_dir', type=str, default=out_dir_default)
parser.add_argument('--eval_interval', type=int, default=eval_interval_default)
parser.add_argument('--log_interval', type=int, default=log_interval_default)
parser.add_argument('--eval_iters', type=int, default=eval_iters_default)
parser.add_argument('--eval_only', action='store_true', default=eval_only_default) # store_true for bools
parser.add_argument('--always_save_checkpoint', action='store_true', default=always_save_checkpoint_default)
parser.add_argument('--init_from', type=str, default=init_from_default)
parser.add_argument('--wandb_log', action='store_true', default=wandb_log_default)
parser.add_argument('--wandb_project', type=str, default=wandb_project_default)
parser.add_argument('--wandb_run_name', type=str, default=wandb_run_name_default)
parser.add_argument('--dataset', type=str, default=dataset_default)
parser.add_argument('--gradient_accumulation_steps', type=int, default=gradient_accumulation_steps_default)
parser.add_argument('--batch_size', type=int, default=batch_size_default)
parser.add_argument('--train_seq_length', type=int, default=train_seq_length_default)
parser.add_argument('--n_layer', type=int, default=n_layer_default)
parser.add_argument('--n_head', type=int, default=n_head_default)
parser.add_argument('--n_embd', type=int, default=n_embd_default)
parser.add_argument('--dropout', type=float, default=dropout_default)
parser.add_argument('--bias', action='store_true', default=bias_default) # if default is False, store_true makes it True if flag is present
parser.add_argument('--no-bias', action='store_false', dest='bias') # to set bias to False if default is True
parser.add_argument('--attention_type', type=str, default=attention_type_default, choices=['full', 'nsa'])
parser.add_argument('--nsa_block_size', type=int, default=nsa_block_size_default)
parser.add_argument('--nsa_topk', type=int, default=nsa_topk_default)
parser.add_argument('--nsa_num_kv_heads', type=int, default=nsa_num_kv_heads_default)
parser.add_argument('--nsa_window_size', type=int, default=nsa_window_size_default)
parser.add_argument('--nsa_local_blocks', type=int, default=nsa_local_blocks_default)
parser.add_argument('--rope_enable', action='store_true', default=rope_enable_default)
parser.add_argument('--learning_rate', type=float, default=learning_rate_default)
parser.add_argument('--max_iters', type=int, default=max_iters_default)
parser.add_argument('--weight_decay', type=float, default=weight_decay_default)
parser.add_argument('--beta1', type=float, default=beta1_default)
parser.add_argument('--beta2', type=float, default=beta2_default)
parser.add_argument('--grad_clip', type=float, default=grad_clip_default)
parser.add_argument('--decay_lr', action='store_true', default=decay_lr_default)
parser.add_argument('--no-decay_lr', action='store_false', dest='decay_lr')
parser.add_argument('--warmup_iters', type=int, default=warmup_iters_default)
parser.add_argument('--lr_decay_iters', type=int, default=lr_decay_iters_default)
parser.add_argument('--min_lr', type=float, default=min_lr_default)
parser.add_argument('--backend', type=str, default=backend_default)
parser.add_argument('--device', type=str, default=device_default)
parser.add_argument('--dtype', type=str, default=dtype_default, choices=['float32', 'bfloat16', 'float16'])
parser.add_argument('--compile', action='store_true', default=compile_default)
parser.add_argument('--no-compile', action='store_false', dest='compile')
parser.add_argument('--ruler_eval_enabled', action='store_true', default=ruler_eval_enabled_default)
parser.add_argument('--ruler_eval_dir', type=str, default=ruler_eval_dir_default)
parser.add_argument('--ruler_eval_interval', type=int, default=ruler_eval_interval_default)
parser.add_argument('--ruler_samples_per_task', type=int, default=ruler_samples_per_task_default)
parser.add_argument('--ruler_eval_max_new_tokens', type=int, default=ruler_eval_max_new_tokens_default)

args = parser.parse_args()

# Update globals with parsed arguments
# This makes the rest of the script work as before, using these global variables
# Alternatively, pass `args` around or convert it to a dict.
out_dir = args.out_dir
eval_interval = args.eval_interval
log_interval = args.log_interval # log_interval_default was not used, but keeping pattern
eval_iters = args.eval_iters
eval_only = args.eval_only
always_save_checkpoint = args.always_save_checkpoint
init_from = args.init_from
wandb_log = args.wandb_log
wandb_project = args.wandb_project
wandb_run_name = args.wandb_run_name
dataset = args.dataset
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size = args.batch_size
train_seq_length = args.train_seq_length
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
attention_type = args.attention_type
nsa_block_size = args.nsa_block_size
nsa_topk = args.nsa_topk
nsa_num_kv_heads = args.nsa_num_kv_heads
nsa_window_size = args.nsa_window_size
nsa_local_blocks = args.nsa_local_blocks
rope_enable = args.rope_enable
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
grad_clip = args.grad_clip
decay_lr = args.decay_lr
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
min_lr = args.min_lr
backend = args.backend
device = args.device
dtype = args.dtype
compile = args.compile
ruler_eval_enabled = args.ruler_eval_enabled
ruler_eval_dir = args.ruler_eval_dir
ruler_eval_interval = args.ruler_eval_interval
ruler_samples_per_task = args.ruler_samples_per_task
ruler_eval_max_new_tokens = args.ruler_eval_max_new_tokens

# Create a config dictionary for logging and checkpointing
# This captures the final effective configuration
config_keys_for_logging = [
    'out_dir', 'eval_interval', 'log_interval', 'eval_iters', 'eval_only',
    'always_save_checkpoint', 'init_from', 'wandb_log', 'wandb_project',
    'wandb_run_name', 'dataset', 'gradient_accumulation_steps', 'batch_size',
    'train_seq_length', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias',
    'attention_type', 'nsa_block_size', 'nsa_topk', 'nsa_num_kv_heads',
    'nsa_window_size', 'nsa_local_blocks', 'rope_enable', 'learning_rate',
    'max_iters', 'weight_decay', 'beta1', 'beta2', 'grad_clip', 'decay_lr',
    'warmup_iters', 'lr_decay_iters', 'min_lr', 'backend', 'device', 'dtype',
    'compile', 'ruler_eval_enabled', 'ruler_eval_dir', 'ruler_eval_interval',
    'ruler_samples_per_task', 'ruler_eval_max_new_tokens'
]
config = {k: globals()[k] for k in config_keys_for_logging}
# -----------------------------------------------------------------------------

# DDP and Slurm integration
use_slurm = os.environ.get('SLURM_JOB_ID') is not None
if use_slurm:
    rank_env = os.environ.get('SLURM_PROCID')
    world_size_env = os.environ.get('SLURM_NTASKS')
    local_rank_env = os.environ.get('SLURM_LOCALID')

    if rank_env is None or world_size_env is None or local_rank_env is None:
        print("Warning: Slurm environment detected, but one or more required Slurm variables are missing.")
        print("SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID must be set.")
        print("Falling back to torchrun/manual DDP environment variables if available.")
    else:
        rank = int(rank_env)
        world_size = int(world_size_env)
        local_rank = int(local_rank_env)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        if 'MASTER_ADDR' not in os.environ:
            master_node_env = os.environ.get('SLURM_NODELIST')
            if master_node_env:
                master_node = master_node_env.split(',')[0].split('(')[0]
                os.environ['MASTER_ADDR'] = master_node
            else:
                print("Warning: SLURM_NODELIST not found, MASTER_ADDR cannot be auto-configured for Slurm.")
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12910' # A default port, can be changed
        print(f"Slurm integration: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Check if essential DDP environment variables are set
    if any(os.environ.get(var) is None for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']):
        print("Error: DDP run indicated, but one or more required DDP environment variables are missing.")
        print("Ensure RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT are set.")
        print("If using Slurm, ensure Slurm variables are correctly translated or set them in your sbatch script.")
        exit(1) # Exit if DDP setup is incomplete

    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    if gradient_accumulation_steps % ddp_world_size != 0:
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is not divisible by ddp_world_size ({ddp_world_size}).")
        print("This might lead to uneven load distribution or errors. Consider adjusting.")
    gradient_accumulation_steps //= ddp_world_size # scale down grad accum steps
    if gradient_accumulation_steps == 0 : gradient_accumulation_steps = 1 # ensure it's at least 1
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * train_seq_length
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - train_seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+train_seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+train_seq_length]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
# All model_args are now directly from the global scope (set by argparse or defaults)
model_args_dict = {
    'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd, 'train_seq_length': train_seq_length,
    'bias': bias, 'vocab_size': None, 'dropout': dropout, 'attention_type': attention_type,
    'nsa_block_size': nsa_block_size, 'nsa_topk': nsa_topk,
    'nsa_num_kv_heads': nsa_num_kv_heads, 'nsa_window_size': nsa_window_size,
    'nsa_local_blocks': nsa_local_blocks, 'rope_enable': rope_enable
}

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args_dict['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args_dict)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force essential architecture attributes to be equal to checkpoint
    arch_keys = ['n_layer', 'n_head', 'n_embd', 'train_seq_length', 'bias', 'vocab_size', 'attention_type',
                 'nsa_block_size', 'nsa_topk', 'nsa_num_kv_heads', 'nsa_window_size', 'nsa_local_blocks', 'rope_enable']
    for k in arch_keys:
        if k in checkpoint_model_args: # for backwards compatibility with old checkpoints
            model_args_dict[k] = checkpoint_model_args[k]
        elif hasattr(args, k): # If not in checkpoint, but in current args, use current args
             model_args_dict[k] = getattr(args, k)

    gptconf = GPTConfig(**model_args_dict)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k_ckpt,v_ckpt in list(state_dict.items()):
        if k_ckpt.startswith(unwanted_prefix):
            state_dict[k_ckpt[len(unwanted_prefix):]] = state_dict.pop(k_ckpt)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args_gpt2 = dict(dropout=dropout) # Pass current dropout to from_pretrained
    model = GPT.from_pretrained(init_from, override_args_gpt2)
    # read off the created config params from the loaded model, so we can store them into checkpoint correctly
    for k_model in ['n_layer', 'n_head', 'n_embd', 'train_seq_length', 'bias', 'vocab_size']:
        model_args_dict[k_model] = getattr(model.config, k_model)
    # Ensure current attention_type and related params are part of model_args_dict for saving
    model_args_dict['attention_type'] = attention_type
    # ... (and other NSA/RoPE params if they were meant to override GPT-2 defaults)

# crop down the model block size if desired, using model surgery
#if train_seq_length < model.config.train_seq_length:
#    model.crop_train_seq_length(train_seq_length)
#    model_args_dict['train_seq_length'] = train_seq_length # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint:  # Ensure optimizer state exists
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Initialize tokenizer for RULER evaluation
ruler_tokenizer_enc = tiktoken.get_encoding("gpt2")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                # Pass None for past_key_values during loss estimation
                logits, loss, _ = model(X, targets=Y, past_key_values=None)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- RULER EVALUATION FUNCTIONS ---
def load_ruler_task_samples(ruler_dir, samples_per_task):
    """Loads a specified number of samples from each RULER JSONL task file in a directory."""
    all_task_samples = {}
    if not os.path.isdir(ruler_dir):
        print(f"Warning: RULER evaluation directory not found: {ruler_dir}")
        return all_task_samples

    jsonl_files = glob.glob(os.path.join(ruler_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in RULER directory: {ruler_dir}")
        return all_task_samples

    for file_path in jsonl_files:
        task_name = os.path.splitext(os.path.basename(file_path))[0]
        task_samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= samples_per_task:
                        break
                    try:
                        task_samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Skipping line in {file_path} due to JSON decode error: {e} (Line: {line.strip()})")
            if task_samples:
                all_task_samples[task_name] = task_samples
        except Exception as e:
            print(f"Warning: Could not read or process RULER file {file_path}: {e}")
    return all_task_samples


@torch.no_grad()
def evaluate_ruler_tasks(model_to_eval, tasks_data, tokenizer_enc, device_to_use, max_tokens_gen):
    """
    Generates responses for RULER tasks and calculates accuracy.
    Returns a dictionary of metrics.
    """
    model_to_eval.eval()
    metrics = {}
    total_correct = 0
    total_samples_evaluated = 0

    print("\n--- Starting RULER Task Evaluation ---")

    for task_name, samples in tasks_data.items():
        if not samples:
            print(f"Task: {task_name} - No samples loaded, skipping.")
            continue

        print(f"\nEvaluating Task: {task_name} ({len(samples)} samples)")
        task_correct_count = 0
        for i, item in enumerate(samples):
            prompt = item.get("input", "")  # RULER uses "input" for the prompt
            expected_outputs = item.get("outputs", [])  # RULER "outputs" is a list of target strings

            if not prompt or not expected_outputs:
                print(f"  Sample {i + 1}: Skipping due to missing prompt or expected_outputs.")
                continue

            start_ids = tokenizer_enc.encode(prompt, allowed_special={"<|endoftext|>"})
            x = (torch.tensor(start_ids, dtype=torch.long, device=device_to_use)[None, ...])

            with ctx:  # Use the global mixed-precision context
                y = model_to_eval.generate(x, max_tokens_gen, temperature=0.1, top_k=1)  # Use deterministic generation
                full_response = tokenizer_enc.decode(y[0].tolist())
                # Extract only the newly generated part
                generated_text = full_response[len(prompt):] if len(full_response) > len(prompt) else ""

            # Check if any of the expected outputs are in the generated text
            # This matches the "fully contains the elements of outputs" criteria
            is_correct = False
            for expected_out_item in expected_outputs:
                if str(expected_out_item) in generated_text:  # Ensure expected_out_item is a string
                    is_correct = True
                    break

            if is_correct:
                task_correct_count += 1

            # Optional: print individual sample results for debugging
            print(f"  Sample {i+1}:")
            print(f"    Prompt: {prompt[:100]}...") # Print truncated prompt
            print(f"    Generated: {generated_text[:100]}...")
            print(f"    Expected: {expected_outputs}")
            print(f"    Correct: {is_correct}")

        task_accuracy = (task_correct_count / len(samples)) * 100 if samples else 0
        metrics[f"ruler/{task_name}_accuracy"] = task_accuracy
        print(f"  Task: {task_name} - Accuracy: {task_accuracy:.2f}% ({task_correct_count}/{len(samples)})")

        total_correct += task_correct_count
        total_samples_evaluated += len(samples)

    overall_accuracy = (total_correct / total_samples_evaluated) * 100 if total_samples_evaluated > 0 else 0
    metrics["ruler/overall_accuracy"] = overall_accuracy
    print(f"\n--- RULER Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples_evaluated}) ---")
    model_to_eval.train()
    return metrics


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)  # Fixed division by warmup_iters + 1
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr, "mfu": running_mfu*100})

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': checkpoint_model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss, 'config': config}
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # --- Run RULER Evaluation at its specific interval ---
    if ruler_eval_enabled and iter_num > 0 and iter_num % ruler_eval_interval == 0 and master_process:

        if ruler_eval_dir and os.path.isdir(ruler_eval_dir):
            # Load a subset of samples from each task file in the directory
            ruler_tasks = load_ruler_task_samples(ruler_eval_dir, ruler_samples_per_task)
            if ruler_tasks:
                # Run evaluation, passing the pre-initialized tokenizer
                ruler_metrics = evaluate_ruler_tasks(raw_model, ruler_tasks, ruler_tokenizer_enc, device,
                                                     ruler_eval_max_new_tokens)
                # Log metrics to wandb if enabled
                if wandb_log:
                    # Add current iteration number to the metrics dict for proper logging
                    wandb.log({"iter": iter_num, **ruler_metrics})
        else:
            print(f"Warning: RULER evaluation enabled but directory '{ruler_eval_dir}' not found.")

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, _ = model(X, targets=Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            # MFU estimation is not implemented in the provided model.py, so we comment it out.
            # You can re-enable it if your model class has an `estimate_mfu` method.
            # mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            mfu_str = "" # f", mfu {running_mfu*100:.2f}%"
        else:
            mfu_str = ""
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms{mfu_str}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
