"""
Sample from a trained model using inputs from JSONL files in a directory
"""
import os
import json
import pickle
from contextlib import nullcontext

import torch
import tiktoken
from model import GPTConfig, GPT
import logging

# Directory containing the JSONL files
data_directory = 'ruler_tasks/4k'  # Replace with your directory path

# Number of samples to process from each file
samples_per_file = 1  # Adjust as needed

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
max_new_tokens = 5  # number of tokens generated in each sample
temperature = 0.8  # temperature for sampling
top_k = 1  # greedy decoding
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'  # 'float32', 'bfloat16', or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster

pe = 'rope'  # examples: 'abs', 'rope', 'alibi', 'nope'
flash = False  # examples: 'True', 'False'
loglevel = 'info'
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

loglevel = {'debug': logging.DEBUG, 'warning': logging.WARNING, 'info': logging.INFO,
            'error': logging.ERROR, 'critical': logging.CRITICAL}[loglevel]
logging.basicConfig(level=loglevel)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

collect_info = False

# Load the model
if init_from == 'resume':
    # Load from a saved checkpoint
    ckpt_path = "ngpt_120M_rope_nonfa_scale_0_100_locheads9_rand_topk10.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Update model arguments if necessary
    checkpoint['model_args']['precision'] = 'float32'
    checkpoint['model_args']['flash'] = flash
    checkpoint['model_args']['block_size'] = 10000  # Set a large block size to accommodate long inputs
    checkpoint['model_args']['self_extend'] = True
    checkpoint['model_args']['use_pseudo_flash'] = True
    checkpoint['model_args']['pseudo_flash_chunk_size'] = 4096

    logging.info(f"{pe} {flash}")

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)

elif init_from.startswith('gpt2'):
    # Initialize from a GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model = model.to(device)
model = model.to(dtype=ptdtype)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Set up the encoding and decoding functions
load_meta = False  # Set to True if you have a meta.pkl file with tokenizer info
if load_meta:
    logging.info(f"Loading meta from meta.pkl...")
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # Use GPT-2 tokenizer
    logging.info("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Collect all JSONL files in the directory
jsonl_files = [f for f in os.listdir(data_directory) if f.endswith('.jsonl')]
print(f"Found {len(jsonl_files)} JSONL files in '{data_directory}':")
for filename in jsonl_files:
    print(f" - {filename}")

# Process each JSONL file
for filename in jsonl_files:
    filepath = os.path.join(data_directory, filename)
    print(f"\nProcessing file '{filename}'")
    total_samples = 0
    correct_predictions = 0

    # First, read all lines from the JSONL file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")

    # Limit the number of samples if necessary
    num_samples = min(samples_per_file, len(lines))
    print(f"Processing {num_samples} samples from the file.")

    for idx, line in enumerate(lines[:num_samples]):
        data = json.loads(line)
        start = data['input']
        expected_outputs = data['outputs']
        index = data.get('index', idx)

        # Encode the input
        start_ids = encode(start)
        input_length = len(start_ids)
        total_length = input_length + max_new_tokens

        # Print input information
        print(f"\nSample index: {index}")
        print(f"Input length (tokens): {input_length}")
        print(f"Expected outputs: {expected_outputs}")

        # Get starting 50 and ending 50 tokens of the input
        start_tokens = start_ids[:50]
        end_tokens = start_ids[-50:] if input_length >= 50 else start_ids
        start_text = decode(start_tokens)
        end_text = decode(end_tokens)

        print(f"Starting 50 tokens of the input:\n{start_text}")
        print(f"Ending 50 tokens of the input:\n{end_text}")

        if total_length > model.config.block_size:
            print(f"Input too long for model (length {total_length}), skipping this sample.")
            continue

        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Run generation
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decode=decode)
                out = decode(y[0].tolist())
                generated_output = out[len(start):].strip()

                # Compare generated output to expected outputs
                match = any(expected_output in generated_output for expected_output in expected_outputs)
                if match:
                    print(f"Correct output for sample index {index}")
                    correct_predictions += 1
                else:
                    print(f"Incorrect output for sample index {index}")

                # Print the generated output and expected outputs
                print(f"Generated output:\n{generated_output}")
                print(f"Expected outputs: {expected_outputs}")
                print('---------------')

        total_samples += 1

    # Compute and display accuracy for the file
    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        print(f"\nAccuracy for file '{filename}': {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    else:
        print(f"No samples were processed for file '{filename}'.")
