"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
from tabnanny import check

import torch
import tiktoken
from model import GPTConfig, GPT
import logging


instr = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
distractor_unit = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
# print(f'distractor len: {len(distractor)}')
needle = "One of the special magic numbers for jobless-speech is: 8090293."
query = "What is the special magic number for jobless-speech mentioned in the provided text? The special magic number for jobless-speech mentioned in the provided text is"
#n_distractors = 160#4k
n_distractors = 600
distractor = ' '.join([distractor_unit] * n_distractors)
print(f'distractor len: {len(distractor)}')
needle_pos_within_distractor = 100

distractor_needle = distractor[:needle_pos_within_distractor] + " " + needle + " " + distractor[needle_pos_within_distractor:]

# print(f'distractor_needle: {distractor_needle}')

start = " ".join([instr, distractor_needle, query])
start += ":"
print(f'start len: {len(start)}, start: {start}')
MAX_SEQ_LEN = 4096
MAX_SEQ_LEN = 20000
#MAX_SEQ_LEN = 40000#4096 # override what's in the checkpoint

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
#start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 3 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
#top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
top_k = 1 # greedy
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# Dima
pe = 'rope' # examples: 'abs', 'rope', 'alibi', 'nope'
flash = False # examples: 'True', 'False'
loglevel = 'info'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

loglevel = {'debug': logging.DEBUG, 'warning': logging.WARNING, 'info': logging.INFO, 'error': logging.ERROR, 'critical': logging.CRITICAL}[loglevel]
logging.basicConfig(level=loglevel)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

collect_info = False

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    #ckpt_path = "gpt2-rope.pt"
    #info_path = "gpt2-rope.info"

    #ckpt_path = "ngpt_120M_rope_nonfa_logscale_k0_5_val306.pt"
    #info_path = "ngpt_120M_rope_nonfa_val303.info"

    #ckpt_path = "ngpt_120M_rope_nonfa_logscale_k0_5_val303.pt"
    #info_path = "ngpt_120M_rope_nonfa_logscale_k0_5_val303.info"

    #ckpt_path = "ngpt_120M_rope_nonfa_scale4k_val293.pt"
    #info_path = "ngpt_120M_rope_nonfa_scale4k_val293.info"

    #ckpt_path = "ngpt_120M_rope_nonfa_val299.pt"
    #info_path = "ngpt_120M_rope_nonfa_val299.info"

    ckpt_path = "ngpt_120M_rope_nonfa_relu_topk10_val308.pt"
    #info_path = "ngpt_120M_rope_nonfa_relu_topk10_val308_4k.info"
    #info_path = "ngpt_120M_rope_nonfa_relu_topk10_val308_n_distr_" + str(n_distractors) + ".info"

    #ckpt_path = "ngpt_120M_rope_nonfa_topk10_val308.pt"

    #ckpt_path = "ngpt_120M_rope_nonfa_relu_val298.pt"

    #ckpt_path = "ngpt_120M_rope_nonfa_min_p_01_val306.pt"

    #ckpt_path = "ngpt_120M_rope_nonfa_locheads9_rand_val302.pt"
    #ckpt_path = "ngpt_120M_rope_nonfa_locheads9_rand_relu_val308.pt"
    #ckpt_path = "ngpt_120M_rope_nonfa_locheads9_rand_topk10_val319.pt"

    #ckpt_path = "ngpt_120M_rope_nonfa_silu_val303.pt"


    checkpoint = torch.load(ckpt_path, map_location=device)

    #checkpoint['model_args']['pe'] = pe
    checkpoint['model_args']['precision'] = 'float16'
    checkpoint['model_args']['flash'] = flash
    checkpoint['model_args']['block_size'] = MAX_SEQ_LEN

    #checkpoint['model_args']['scaling_target_sequence_length'] = MAX_SEQ_LEN
    #checkpoint['model_args']['use_nGPT'] = True
    checkpoint['model_args']['self_extend'] = True

    #checkpoint['model_args']['relu_before_attn_softmax'] = True

    #checkpoint['model_args']['rope_base'] = 100000

    #checkpoint['model_args']['softmax_log_k'] = 0.1
    #checkpoint['model_args']['relu_instead_of_attn_softmax'] = True
    checkpoint['model_args']['topk_after_attn_softmax'] = 10

    #checkpoint['model_args']['window_size'] = 512

    logging.info(f"{pe} {flash}")

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)

elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    logging.info(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    logging.info("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
assert len(start_ids) + max_new_tokens <= model.config.block_size, \
    f"Model max seq len: {model.config.block_size}, but passed len start_ids: {len(start_ids)} and max_new_tokens: {max_new_tokens}"

logging.info(f"Model max seq len: {model.config.block_size}; seeing len start_ids: {len(start_ids)} and max_new_tokens: {max_new_tokens}")
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)



# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if collect_info:
                y, info, _ = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    collect_info=collect_info,
                    collect_probs_per_layer=True,
                    decode=decode
                )
                out = decode(y[0].tolist())

                logging.info(len(y[0].tolist()))
                logging.info(len(info))
                # should be 1 less since last token doesn't have attention info

                logging.info(f'input: "{start}"')
                logging.info(f'input end: "{out[len(start) - 100:len(start)]}"')
                logging.info(f'output: "{out[len(start):]}"')



                # Decode the main tokens
                """for i in info:
                    i["decoded_token"] = decode([i["token_id"]])  # Assuming decode returns a list

                # Decode the next_token_probs
                for i in info:
                    if "next_token_probs" in i and i["next_token_probs"]:
                        decoded_next_token_probs = []
                        for token_id, prob in i["next_token_probs"]:
                            decoded_token = decode([token_id])  # Decode each token_id
                            decoded_next_token_probs.append({
                                "token_id": token_id,
                                "decoded_token": decoded_token,
                                "probability": prob
                            })
                        i["next_token_probs"] = decoded_next_token_probs
                """

                text = [info[i]["decoded_token"] for i in range(len(info) - 5 - max_new_tokens, len(info))]

                logging.info(f'len(text): {len(text)}')
                logging.info(f'text: {text}')

                torch.save(info, info_path)

                logging.info('---------------')
            else:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decode=decode)
                out = decode(y[0].tolist())
                logging.info(f'input: "{start}"')
                logging.info(f'input end: "{out[len(start)-100:len(start)]}"')
                logging.info(f'output: "{out[len(start):]}"')
                logging.info('---------------')
