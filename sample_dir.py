"""
Sample from trained models using inputs from JSONL files in a directory
"""

import os
import json
import pickle
from contextlib import nullcontext
from collections import defaultdict
import pandas as pd  # Added for Excel file creation

import torch
import tiktoken
from model import GPTConfig, GPT
import logging

# Directory containing the JSONL files
data_directory = 'ruler_tasks/64k'  # Replace with your directory path

# Number of samples to process from each file
samples_per_file = 1  # Adjust as needed

# List of checkpoint file paths
checkpoint_paths = [
    # "ngpt_120M_rope_nonfa_val299.pt",
    # "ngpt_120M_rope_nonfa_relu_min_p_01.pt",
    # "ngpt_120M_rope_nonfa_scale_0_100_locheads9_rand_topk10.pt",
    #"ngpt_120M_rope_nonfa_locheads9_rand_min_p_02_2.pt",
    # "ngpt_120M_rope_nonfa_relu_min_p_01.pt",
    # "ngpt_120M_rope_nonfa_min_p_01_val306.pt",
    # "ngpt_120M_rope_nonfa_locheads9_rand_val302.pt",
    # "ngpt_120M_rope_nonfa_locheads9_rand_topk10_val319.pt",
    #"ngpt_120M_rope_nonfa_nowd_2.pt",
    #"ngpt_120M_nope_nonfa_nowd_2.pt",
    "ngpt_120M_rope_nonfa_locheads9_rand_min_p_02_nowd_2.pt",
    #"ngpt_120M_nope_nonfa_rand_min_p_02_nowd_2.pt",
    #"ngpt_120M_nope_nonfa_locheads9_rand_min_p_02_nowd_2.pt",
    #"ngpt_120M_rope_nonfa_locheads9_rand_win16_min_p_02_nowd_2.pt",
]

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or a GPT-2 variant (e.g., 'gpt2-xl')
out_dir = 'out'  # Ignored if init_from is not 'resume'
# Removed global max_new_tokens
temperature = 1.0  # Temperature for sampling
top_k = 1  # Use top-k sampling (1 for greedy decoding)
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
precision = 'float32'  # 'float32', 'bfloat16', or 'float16'
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
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[precision]
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

# Function to find positions of expected output in input
def find_sublist_positions(lst, sublst):
    positions = []
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i:i+len(sublst)] == sublst:
            positions.append(i)
    return positions

# Initialize a results dictionary
results = defaultdict(lambda: defaultdict(dict))  # results[model_name][file_name] = {...}

# Process each checkpoint (model)
for ckpt_path in checkpoint_paths:
    model_name = os.path.basename(ckpt_path)
    print(f"\nLoading model from checkpoint '{ckpt_path}'")

    # Load the model
    if init_from == 'resume':
        # Load from a saved checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Update model arguments if necessary
        checkpoint['model_args']['precision'] = precision#'float32'
        checkpoint['model_args']['flash'] = flash
        checkpoint['model_args']['block_size'] = 70000  # Set a large block size to accommodate long inputs
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
        total_input_token_length = 0  # Sum of input token lengths
        total_expected_output_token_length = 0  # Sum of expected output token lengths

        # For position statistics
        positions_all_samples = []
        positions_correct_predictions = []

        # For confidence tracking
        first_token_confidences = []
        max_first_token_confidences_in_incorrect_samples = []

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
            if start and start[-1] != ':':
                start += ':'

            expected_outputs = data['outputs']
            # Include versions of expected outputs with a leading space
            expected_outputs_with_space = [" " + output for output in expected_outputs]
            expected_outputs.extend(expected_outputs_with_space)
            index = data.get('index', idx)

            # Encode the input
            start_ids = encode(start)
            input_length = len(start_ids)
            total_input_token_length += input_length

            # Compute max token length of expected outputs
            expected_output_lengths = [len(encode(output)) for output in expected_outputs]
            max_expected_output_length = max(expected_output_lengths) if expected_output_lengths else 0
            total_expected_output_token_length += sum(expected_output_lengths) / len(expected_output_lengths) if expected_output_lengths else 0

            # Set max_new_tokens for this sample
            max_new_tokens = max_expected_output_length

            total_length = input_length + max_new_tokens

            # Print input information
            print(f"\nSample index: {index}")
            print(f"Input length (tokens): {input_length}")
            print(f"Expected outputs: {expected_outputs}")
            print(f"Max expected output length (tokens): {max_expected_output_length}")
            print(f"max_new_tokens for this sample: {max_new_tokens}")

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
                    # Modify the generate call to receive next_token_prob_list
                    y, next_token_prob_list = model.generate(
                        x,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        decode=decode,
                    )
                    out = decode(y[0].tolist())
                    generated_output = out[len(start):]

                    # Token match metric
                    max_token_match = 0.0
                    for expected_output in expected_outputs:
                        expected_tokens = encode(expected_output)
                        generated_tokens = encode(generated_output)
                        if expected_tokens:
                            match_count = sum(1 for et, gt in zip(expected_tokens, generated_tokens) if et == gt)
                            token_match_percentage = match_count / len(expected_tokens)
                        else:
                            token_match_percentage = 0.0
                        if token_match_percentage > max_token_match:
                            max_token_match = token_match_percentage

                    if max_token_match == 1.0:
                        print(f"Correct output for sample index {index}")
                        correct_predictions += 1
                    else:
                        print(f"Incorrect output for sample index {index}")

                    total_token_match += max_token_match
                    total_possible_token_match += 1

                    # Find expected output positions in input for all expected outputs
                    found_positions = False
                    for expected_output in expected_outputs:
                        expected_output = expected_output
                        expected_tokens_in_input = encode(expected_output)
                        positions = find_sublist_positions(start_ids, expected_tokens_in_input)
                        if positions:
                            found_positions = True
                            positions_all_samples.extend(positions)
                            if max_token_match == 1.0:
                                positions_correct_predictions.extend(positions)
                            print(f"Expected output '{expected_output}' found in input for sample index {index} at position {positions}")
                        else:
                            print(f"Expected output '{expected_output}' not found in input for sample index {index}")

                    # Tracking probability of the first correct token
                    # Collect first token IDs of expected outputs
                    expected_first_token_ids = [encode(output)[0] for output in expected_outputs if encode(output)]

                    found_first_token_confidence = None
                    found_first_token = False
                    # Iterate over generated tokens
                    gen_tokens = y[0][len(x[0]):].tolist()
                    for i, gen_token_id in enumerate(gen_tokens):
                        if gen_token_id in expected_first_token_ids:
                            # Get the probability from next_token_prob_list
                            if i < len(next_token_prob_list):
                                next_probs = next_token_prob_list[i]
                                # Find the probability of the generated token
                                for prob_info in next_probs:
                                    if prob_info['token_id'] == gen_token_id:
                                        found_first_token_confidence = prob_info['probability']
                                        first_token_confidences.append(found_first_token_confidence)
                                        print(f"First correct token probability: {found_first_token_confidence}")
                                        break
                            found_first_token = True
                            break
                    if not found_first_token:
                        print(f"First expected token not found in generated output for sample index {index}")
                        # For this sample, find the maximum probability assigned to any starting token of expected outputs in top 10 at any position
                        max_confidence = 0.0
                        for next_probs in next_token_prob_list:
                            for prob_info in next_probs:
                                if prob_info['token_id'] in expected_first_token_ids:
                                    if prob_info['probability'] > max_confidence:
                                        max_confidence = prob_info['probability']
                            # Early exit if max_confidence is 1.0
                            if max_confidence == 1.0:
                                break
                        if max_token_match != 1.0:
                            # Only collect for incorrect samples
                            max_first_token_confidences_in_incorrect_samples.append(max_confidence)
                        print(f"Max confidence in starting expected tokens in incorrect sample index {index}: {max_confidence}")

                    # Print the generated output and expected outputs
                    print(f"Generated output:\n{generated_output}")
                    print(f"Token match percentage: {max_token_match * 100:.2f}%")
                    print('---------------')

            total_samples += 1

        # Compute position statistics
        def compute_position_stats(positions_list):
            if positions_list:
                min_pos = min(positions_list)
                avg_pos = sum(positions_list) / len(positions_list)
                max_pos = max(positions_list)
                return min_pos, avg_pos, max_pos
            else:
                return None, None, None

        min_pos_all, avg_pos_all, max_pos_all = compute_position_stats(positions_all_samples)
        min_pos_correct, avg_pos_correct, max_pos_correct = compute_position_stats(positions_correct_predictions)

        # Compute average confidence in first correct token
        if first_token_confidences:
            average_confidence = sum(first_token_confidences) / len(first_token_confidences)
        else:
            average_confidence = None

        # Compute average max confidence in starting expected tokens in incorrect samples
        if max_first_token_confidences_in_incorrect_samples:
            average_max_confidence_in_incorrect = sum(max_first_token_confidences_in_incorrect_samples) / len(max_first_token_confidences_in_incorrect_samples)
        else:
            average_max_confidence_in_incorrect = None

        # Compute and store accuracy and token match metrics for the file
        if total_samples > 0:
            accuracy = (correct_predictions / total_samples) * 100
            average_token_match = (total_token_match / total_possible_token_match) * 100
            average_input_token_length = total_input_token_length / total_samples
            average_expected_output_token_length = total_expected_output_token_length / total_samples

            results[model_name][filename] = {
                'Model': model_name,
                'File': filename,
                'Accuracy (%)': accuracy,
                'Correct Predictions': correct_predictions,
                'Total Samples': total_samples,
                'Average Token Match (%)': average_token_match,
                'Average Input Token Length': average_input_token_length,
                'Average Expected Output Token Length': average_expected_output_token_length,
                'Min Position (All Samples)': min_pos_all,
                'Avg Position (All Samples)': avg_pos_all,
                'Max Position (All Samples)': max_pos_all,
                'Min Position (Correct Predictions)': min_pos_correct,
                'Avg Position (Correct Predictions)': avg_pos_correct,
                'Max Position (Correct Predictions)': max_pos_correct,
                'Average Confidence in First Correct Token': average_confidence,
                'Average Max Confidence in Incorrect Samples': average_max_confidence_in_incorrect,
            }
            print(f"\nResults for model '{model_name}' on file '{filename}':")
            print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
            print(f"Average token match: {average_token_match:.2f}%")
            print(f"Average input token length: {average_input_token_length:.2f}")
            print(f"Average expected output token length: {average_expected_output_token_length:.2f}")
            print(f"Expected output positions in input (all samples): min={min_pos_all}, avg={avg_pos_all}, max={max_pos_all}")
            print(f"Expected output positions in input (correct predictions): min={min_pos_correct}, avg={avg_pos_correct}, max={max_pos_correct}")
            print(f"Average confidence in first correct token: {average_confidence}")
            print(f"Average max confidence in starting expected tokens in incorrect samples: {average_max_confidence_in_incorrect}")
        else:
            print(f"No samples were processed for file '{filename}' with model '{model_name}'.")

# After all models and files are processed, print concise stats
print("\nFinal Results Summary:")
# Prepare data for Excel file
excel_data = []

for model_name in results:
    print(f"\nModel: {model_name}")
    for filename in results[model_name]:
        res = results[model_name][filename]
        print(f"  File: {filename}")
        print(f"    Accuracy: {res['Accuracy (%)']:.2f}% ({res['Correct Predictions']}/{res['Total Samples']})")
        print(f"    Average token match: {res['Average Token Match (%)']:.2f}%")
        print(f"    Average input token length: {res['Average Input Token Length']:.2f}")
        print(f"    Average expected output token length: {res['Average Expected Output Token Length']:.2f}")
        print(f"    Expected output positions in input (all samples): min={res['Min Position (All Samples)']}, avg={res['Avg Position (All Samples)']}, max={res['Max Position (All Samples)']}")
        print(f"    Expected output positions in input (correct predictions): min={res['Min Position (Correct Predictions)']}, avg={res['Avg Position (Correct Predictions)']}, max={res['Max Position (Correct Predictions)']}")
        print(f"    Average confidence in first correct token: {res['Average Confidence in First Correct Token']}")
        print(f"    Average max confidence in starting expected tokens in incorrect samples: {res['Average Max Confidence in Incorrect Samples']}")
        # Add to excel_data
        excel_data.append(res)

# Get the folder name
folder_name_base = os.path.basename(data_directory)

# Create a DataFrame and save to Excel
#df = pd.DataFrame(excel_data)
#excel_file_name = 'results_summary_' + folder_name_base + '.xlsx'
#df.to_excel(excel_file_name, index=False)
#print(f"\nResults have been saved to '{excel_file_name}'.")
