"""
Sample from trained models using inputs from JSONL files in a directory
"""

import os
import json
import pickle
from contextlib import nullcontext
from collections import defaultdict

import torch
import tiktoken
from model import GPTConfig, GPT
import logging

# Directory containing the JSONL files
data_directory = 'ruler_tasks/single_niah_2'  # Replace with your directory path

# Number of samples to process from each file
samples_per_file = 1  # Adjust as needed

# List of checkpoint file paths
checkpoint_paths = [
    "ngpt_120M_rope_nonfa_val299.pt",
    "ngpt_120M_rope_nonfa_relu_min_p_01.pt",
    "ngpt_120M_rope_nonfa_scale_0_100_locheads9_rand_topk10.pt",
    # Add more checkpoints as needed
]

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or a GPT-2 variant (e.g., 'gpt2-xl')
out_dir = 'out'  # Ignored if init_from is not 'resume'
max_new_tokens = 5  # Number of tokens to generate in each sample
temperature = 0.8  # Temperature for sampling
top_k = 1  # Use top-k sampling (1 for greedy decoding)
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'  # 'float32', 'bfloat16', or 'float16'
compile = False  # Use PyTorch 2.0 to compile the model for faster inference

pe = 'rope'  # Positional encoding type
flash = False  # Whether to use flash attention
loglevel = 'info'
# exec(open('configurator.py').read())  # Uncomment if you have a configurator.py
# -----------------------------------------------------------------------------

loglevel = {'debug': logging.DEBUG, 'warning': logging.WARNING, 'info': logging.INFO,
            'error': logging.ERROR, 'critical': logging.CRITICAL}[loglevel]
logging.basicConfig(level=loglevel)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN
device_type = 'cuda' if 'cuda' in device else 'cpu'  # For torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

collect_info = False

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

# Initialize a results dictionary
results = defaultdict(lambda: defaultdict(dict))  # results[model_name][file_name] = {'accuracy': ..., 'token_match': ...}

# Process each checkpoint (model)
for ckpt_path in checkpoint_paths:
    model_name = os.path.basename(ckpt_path)
    print(f"\nLoading model from checkpoint '{ckpt_path}'")

    # Load the model
    if init_from == 'resume':
        # Load from a saved checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Update model arguments if necessary
        checkpoint['model_args']['precision'] = 'float32'
        checkpoint['model_args']['flash'] = flash
        checkpoint['model_args']['block_size'] = 20000  # Set a large block size to accommodate long inputs
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
        model = torch.compile(model)  # Requires PyTorch 2.0

    # Process each JSONL file
    for filename in jsonl_files:
        filepath = os.path.join(data_directory, filename)
        print(f"\nProcessing file '{filename}' with model '{model_name}'")
        total_samples = 0
        correct_predictions = 0
        total_token_match = 0.0  # Sum of token match percentages
        total_possible_token_match = 0  # Total number of samples (for averaging)

        # First, read all lines from the JSONL file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Total lines in file: {len(lines)}")

        # Limit the number of samples if necessary
        num_samples = min(samples_per_file, len(lines))
        print(f"Processing {num_samples} samples from the file.")

        for idx, line in enumerate(lines[:num_samples]):
            data = json.loads(line)
            start = data['input'].replace('\\n', '\n').replace('<extra_id_0>', '').replace('<extra_id_1>', '')
            if start[-1] != ':': start += ':'

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
                    exact_match = any(expected_output in generated_output for expected_output in expected_outputs)
                    if exact_match:
                        print(f"Correct output for sample index {index}")
                        correct_predictions += 1
                    else:
                        print(f"Incorrect output for sample index {index}")

                    # Token match metric
                    max_token_match = 0.0
                    for expected_output in expected_outputs:
                        expected_tokens = encode(expected_output)
                        generated_tokens = encode(generated_output)
                        match_count = sum(1 for token in expected_tokens if token in generated_tokens)
                        token_match_percentage = match_count / len(expected_tokens) if expected_tokens else 0.0
                        if token_match_percentage > max_token_match:
                            max_token_match = token_match_percentage

                    total_token_match += max_token_match
                    total_possible_token_match += 1

                    # Print the generated output and expected outputs
                    print(f"Generated output:\n{generated_output}")
                    print(f"Token match percentage: {max_token_match * 100:.2f}%")
                    print('---------------')

            total_samples += 1

        # Compute and store accuracy and token match metrics for the file
        if total_samples > 0:
            accuracy = (correct_predictions / total_samples) * 100
            average_token_match = (total_token_match / total_possible_token_match) * 100
            results[model_name][filename] = {
                'accuracy': accuracy,
                'average_token_match': average_token_match,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions
            }
            print(f"\nResults for model '{model_name}' on file '{filename}':")
            print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
            print(f"Average token match: {average_token_match:.2f}%")
        else:
            print(f"No samples were processed for file '{filename}' with model '{model_name}'.")

# After all models and files are processed, print concise stats
print("\nFinal Results Summary:")
for model_name in results:
    print(f"\nModel: {model_name}")
    for filename in results[model_name]:
        res = results[model_name][filename]
        print(f"  File: {filename}")
        print(f"    Accuracy: {res['accuracy']:.2f}% ({res['correct_predictions']}/{res['total_samples']})")
        print(f"    Average token match: {res['average_token_match']:.2f}%")
