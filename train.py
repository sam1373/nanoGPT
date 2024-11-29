"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import sys
import glob
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import GPTConfig, GPT
from datetime import timedelta

import logging


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
loglevel = 'info'
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
use_distributed_data_loader = False
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
pe = 'abs' # examples: 'abs', 'rope', 'alibi', 'nope', 'xpos2'
flash = True # examples: 'True', 'False'
rope_base = 10000 # RoPE base
rope_percentage = 1.0
rope_wavelengths = None
xpos2_decay_base = 2.0 # Decay base
xpos2_decay_angle = math.pi / 2 # Soft max angle
xpos2_adaptive = True # Should we change decay angle if there's risk of overflow
scaling_target_sequence_length = None
softmax_log_k = 0.0
use_nGPT = 0
base_scale = None
relu_instead_of_attn_softmax = False
topk_after_attn_softmax = 0
relu_neg_inf = False
pretraining_seq_length = block_size
window_size = 128
self_extend = False
head_dropout = 0
local_heads_during_training = 0
local_window_size = 128
local_heads_random = False
top_p = 0.0
min_p = 0.0
top_a = 0.0
silu_before_attn_softmax = False
score_threshold = 0.0
score_scale = 1.0
q_constant_scale = 1.0
softmax_like = 'softmax'
softmax_scale = None
modded = False
use_pseudo_flash = False
pseudo_flash_chunk_size = 512

precision = 'float32'

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda'
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

time_limit_seconds = 14400

starting_checkpoint = None

tlaunch = time.time()
print("Current Directory:", os.getcwd())
# the input configurations will overwrite all configs given above!
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

loglevel = {'debug': logging.DEBUG, 'warning': logging.WARNING, 'info': logging.INFO, 'error': logging.ERROR, 'critical': logging.CRITICAL}[loglevel]
logging.basicConfig(level=loglevel)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    #init_process_group(backend=backend)
    dist.init_process_group(backend=backend,
        timeout=timedelta(milliseconds=20*60000) # Setting a 20-minute timeout
    )
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    dist.barrier()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[precision]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = dataset#os.path.join('data', dataset)


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank, num_processes, batch_size):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T
        self.batch_size = batch_size

        # Glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # Load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.current_shard])
        # Calculate the positions assigned to this process
        self.total_positions = len(self.tokens) - self.T - 1
        positions_per_process = self.total_positions // self.num_processes
        self.start_pos = self.process_rank * positions_per_process
        self.end_pos = self.start_pos + positions_per_process
        self.current_position = self.start_pos

    def next_batch(self):
        x_batch = []
        y_batch = []
        for _ in range(self.batch_size):
            # If we have reached the end of our assigned positions, advance to the next shard
            if self.current_position + self.T + 1 > self.end_pos:
                self.advance()
            buf = self.tokens[self.current_position:self.current_position + self.T + 1]
            if len(buf) < self.T + 1:
                # In case the buffer is too small, pad or handle appropriately
                continue  # Skip this iteration or handle as needed
            buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
            x_batch.append(buf[:-1])
            y_batch.append(buf[1:])
            # Move to the next position assigned to this process
            self.current_position += 1
        # Ensure that we have collected enough samples
        if len(x_batch) < self.batch_size:
            # Handle this case, possibly by recursively calling next_batch or adjusting batch_size
            pass  # For simplicity, you can fill the batch with additional samples or handle as needed
        x = torch.stack(x_batch)
        y = torch.stack(y_batch)
        return x.cuda(), y.cuda()

if not use_distributed_data_loader:
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
else:
    train_loader = DistributedDataLoader(data_dir + 'train_*.bin', block_size, ddp_local_rank, ddp_world_size, batch_size)
    val_loader = DistributedDataLoader(data_dir + 'val_*.bin', block_size, ddp_local_rank, ddp_world_size, batch_size)
    def get_batch(split):
        if split == 'train':
            return train_loader.next_batch()
        else:
            return val_loader.next_batch()


    print(
        f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(
        f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    print('=' * 100)

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
    logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    bias=bias,
    pe=pe,
    flash=flash,
    rope_base=rope_base,
    rope_percentage=rope_percentage,
    rope_wavelengths=rope_wavelengths,
    xpos2_decay_base=xpos2_decay_base,
    xpos2_decay_angle=xpos2_decay_angle,
    xpos2_adaptive=xpos2_adaptive,
    scaling_target_sequence_length=scaling_target_sequence_length,
    softmax_log_k=softmax_log_k,
    use_nGPT=use_nGPT,
    base_scale=base_scale,
    topk_after_attn_softmax=topk_after_attn_softmax,
    relu_neg_inf=relu_neg_inf,
    pretraining_seq_length=pretraining_seq_length,
    window_size=window_size,
    self_extend=self_extend,
    head_dropout=head_dropout,
    local_heads_during_training=local_heads_during_training,
    local_window_size=local_window_size,
    local_heads_random=local_heads_random,
    top_p=top_p,
    min_p=min_p,
    top_a=top_a,
    silu_before_attn_softmax=silu_before_attn_softmax,
    score_threshold=score_threshold,
    score_scale=score_scale,
    q_constant_scale=q_constant_scale,
    softmax_like=softmax_like,
    precision=precision,
    softmax_scale=softmax_scale,
    modded=modded,
    use_pseudo_flash=use_pseudo_flash,
    pseudo_flash_chunk_size=pseudo_flash_chunk_size
)
if init_from == 'scratch':
    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    logging.info(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    print("checkpoint_model_args:", checkpoint_model_args)
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    print("model_args:", model_args)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
    else:
        best_val_loss = 1e9
elif init_from.startswith('gpt2'):
    logging.info(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
elif init_from == 'start_from':
    logging.info(f"Initializing from {starting_checkpoint}")
    # initialize from a checkpoint
    checkpoint = torch.load(starting_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    print("checkpoint_model_args:", checkpoint_model_args)
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    print("model_args:", model_args)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
    else:
        best_val_loss = 1e9

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(precision == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    logging.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

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
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# training loop
# X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed

if master_process:
    print("learning_rate: %f" % (learning_rate))
    print("min_lr: %f" % (min_lr))
    print("max_iters: %f" % (max_iters))
    print("lr_decay_iters: %f" % (lr_decay_iters))
    print("warmup_iters: %f" % (warmup_iters))
    print("batch_size: %f" % (batch_size))
    print("gradient_accumulation_steps: %f" % (gradient_accumulation_steps))
    print("block_size: %f" % (block_size))
    print("weight_decay: %f" % (weight_decay))
    print("dropout: %f" % (dropout))
    print("bias: %f" % (bias))
    print("pe: %s" % (pe))
    print("flash: %s" % (flash))
    print("rope_base: %f" % (rope_base))
    print("rope_percentage: %f" % (rope_percentage))
    print("rope_wavelengths: %s" % (rope_wavelengths))
    print("xpos2_decay_base: %f" % (xpos2_decay_base))
    print("xpos2_decay_angle: %f" % (xpos2_decay_angle))
    print("xpos2_adaptive: %s" % (xpos2_adaptive))
    print("scaling_target_sequence_length: %s" % (scaling_target_sequence_length))
    print("softmax_log_k: %f" % (softmax_log_k))
    print("use_nGPT: %f" % (use_nGPT))
    print("base_scale: %s" % (base_scale))
    print("relu_instead_of_attn_softmax: %s" % (relu_instead_of_attn_softmax))
    print("topk_after_attn_softmax: %f" % (topk_after_attn_softmax))
    print("relu_neg_inf: %s" % (relu_neg_inf))
    print("pretraining_seq_length: %f" % (pretraining_seq_length))
    print("window_size: %f" % (window_size))
    print("self_extend: %s" % (self_extend))
    print("head_dropout: %f" % (head_dropout))
    print("local_heads_during_training: %f" % (local_heads_during_training))
    print("local_window_size: %f" % (local_window_size))
    print("local_heads_random: %s" % (local_heads_random))
    print("top_p: %f" % (top_p))
    print("min_p: %f" % (min_p))
    print("top_a: %f" % (top_a))
    print("silu_before_attn_softmax: %s" % (silu_before_attn_softmax))
    print("score_threshold: %f" % (score_threshold))
    print("score_scale: %f" % (score_scale))
    print("q_constant_scale: %f" % (q_constant_scale))
    print("softmax_like: %s" % (softmax_like))
    print("softmax_scale: %s" % (softmax_scale))
    print("modded: %s" % (modded))
    print("precision: %s" % (precision))
    print("batch_size: %f" % (batch_size))



time_spent = time.time() - tlaunch
print(f"Time spent: {time_spent} seconds")
starting_iter_num = iter_num
print("starting_iter_num: %d" % iter_num)

if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    transformer = model.module.transformer
    config = model.module.config
    module = model.module
else:
    transformer = model.transformer
    config = model.config
    module = model


def justnorm(x, idim=-1):
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    return res


def normalize_matrices():
    transformer.wte.weight.data.copy_(justnorm(transformer.wte.weight.data, 1))  # V, n_embd
    module.lm_head.weight.data.copy_(justnorm(module.lm_head.weight.data, 1))  # V, n_embd

    for layer_idx in range(0, config.n_layer):
        block = transformer["h"][layer_idx]

        block.attn.c_attn.weight.data.copy_(justnorm(block.attn.c_attn.weight.data, 1))  # n_proj, n_embd
        #block.key.weight.data.copy_(justnorm(block.key.weight.data, 1))  # n_proj, n_embd
        #block.value.weight.data.copy_(justnorm(block.value.weight.data, 1))  # n_proj, n_embd
        block.attn.c_proj.weight.data.copy_(justnorm(block.attn.c_proj.weight.data, 0))  # n_embd, n_proj

        block.mlp.c_fc.weight.data.copy_(justnorm(block.mlp.c_fc.weight.data, 1))  # n_proj, n_embd
        block.mlp.c_proj.weight.data.copy_(justnorm(block.mlp.c_proj.weight.data, 0))  # n_embd, n_proj


if (use_nGPT == 1):
    normalize_matrices()

while True:
    if (1):
        local_seed = 100 * iter_num + seed_offset  # local_seed should never exceed 2.147e+9 because of np.random.seed, 100 here should be > nworkers
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)
        # if (iter_num % 10 == 0):    # uncomment to make sure different seeds are used
        #    print("iter: %d seed: %d" % (iter_num, local_seed))

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        rng_state_pytorch = torch.get_rng_state()
        rng_state_bytes = rng_state_pytorch.numpy().tobytes()
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")

        # long eval?
        # accuracy and token match

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })

        if always_save_checkpoint:
            if iter_num > starting_iter_num:
                tcheckpointsaving_begin = time.time()
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'config': config,
                    'rng_state_pytorch_bytes': rng_state_bytes,
                    'rng_state_numpy': np.random.get_state()
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print("Checkpoint saving time: %f sec" % (time.time() - tcheckpointsaving_begin))



    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    X, Y = get_batch('train')
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        # .scale(loss).backward()
        loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.6f}, time {dt * 1000:.2f}ms")

    if (use_nGPT == 1):
        normalize_matrices()

    if (iter_num % 100 == 0) and master_process:
        print("lr=%f" % lr)

    if master_process:

        if iter_num >= max_iters:
            finished_fname = out_dir + "/finished"
            finished_file = open(finished_fname, "w")
            finished_file.write("1")
            finished_file.close()

    if (time.time() - tlaunch > time_limit_seconds):
        break

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
time_spent = time.time() - tlaunch
print(f"Time spent: {time_spent} seconds")
if ddp:
    dist.barrier()
    dist.destroy_process_group()
