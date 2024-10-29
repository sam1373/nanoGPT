from flask import Flask, render_template, jsonify, request
import torch
import argparse
import sys

app = Flask(__name__)

# Initialize generated_info as None; it will be loaded based on the command-line argument
generated_info = None

def load_generated_info(info_file):
    global generated_info
    try:
        generated_info = torch.load(info_file, map_location='cpu')
        print(f"Successfully loaded '{info_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{info_file}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{info_file}': {e}")
        sys.exit(1)

def preprocess_generated_info():
    """
    Preprocesses the generated_info to handle empty tokens and calculate total attention received.
    Also assigns display tokens and flags for generated tokens.
    """
    global generated_info
    initial_context_length = None
    for info in generated_info:
        if 'initial_context_length' in info:
            initial_context_length = info['initial_context_length']
            break

    # Preprocess generated_info to handle empty tokens and calculate total attention received
    total_tokens = len(generated_info)
    for idx, info in enumerate(generated_info):
        token_text = info.get('decoded_token', '')
        if not token_text.strip():
            info['display_token'] = '___'
        else:
            info['display_token'] = token_text
        info['index'] = idx  # Store index for reference
        info['is_generated'] = not info.get('is_initial_context', False)

        # Initialize total_attention_received to zero; it will be calculated when data is available
        info['total_attention_received'] = 0.0

        # Ensure that 'total_attention_per_layer_head' is initialized
        if 'total_attention_per_layer_head' not in info:
            info['total_attention_per_layer_head'] = []

    # Determine the maximum number of layers and heads
    num_layers = 0
    num_heads = 0
    for info in generated_info:
        if 'attn_info_per_layer' in info:
            num_layers = max(num_layers, len(info['attn_info_per_layer']))
            for layer_info in info['attn_info_per_layer']:
                num_heads = max(num_heads, len(layer_info.get('q_norms', [])))
            break  # We can determine num_layers and num_heads from the first token that has this data

    # Fill in missing data for tokens that lack attention information
    for info in generated_info:
        # Ensure 'attn_info_per_layer' has entries for all layers
        attn_info_per_layer = info.get('attn_info_per_layer', [])
        while len(attn_info_per_layer) < num_layers:
            attn_info_per_layer.append({})
        info['attn_info_per_layer'] = attn_info_per_layer

        # Ensure 'total_attention_per_layer_head' has entries for all layers and heads
        total_attention_per_layer_head = info.get('total_attention_per_layer_head', [])
        while len(total_attention_per_layer_head) < num_layers:
            total_attention_per_layer_head.append([0.0] * num_heads)
        else:
            # Ensure each layer has the correct number of heads
            for layer_attention in total_attention_per_layer_head:
                if len(layer_attention) < num_heads:
                    layer_attention.extend([0.0] * (num_heads - len(layer_attention)))
        info['total_attention_per_layer_head'] = total_attention_per_layer_head

        # Calculate 'total_attention_received' for overall opacity
        info['total_attention_received'] = sum(
            sum(float(attn) for attn in layer_attention) for layer_attention in info['total_attention_per_layer_head']
        )

@app.route('/')
def index():
    total_tokens = len(generated_info)
    if total_tokens == 0:
        return "No tokens available to display.", 400

    # Determine the maximum number of layers and heads
    num_layers = 0
    num_heads = 0
    for info in generated_info:
        if 'attn_info_per_layer' in info:
            num_layers = max(num_layers, len(info['attn_info_per_layer']))
            for layer_info in info['attn_info_per_layer']:
                num_heads = max(num_heads, len(layer_info.get('q_norms', [])))
            break  # We can determine num_layers and num_heads from the first token that has this data

    tokens = []
    display_index = 0
    index_mapping = {}  # Mapping from display index to actual index in generated_info

    if total_tokens <= 2000:
        for idx, info in enumerate(generated_info):
            tokens.append({
                'display_index': display_index,
                'actual_index': idx,
                'token_id': info['token_id'],
                'decoded_token': info['display_token'],
                'is_generated': info['is_generated']
            })
            index_mapping[display_index] = idx
            display_index += 1
    else:
        # First 1000 tokens
        for idx in range(1000):
            info = generated_info[idx]
            tokens.append({
                'display_index': display_index,
                'actual_index': idx,
                'token_id': info['token_id'],
                'decoded_token': info['display_token'],
                'is_generated': info['is_generated']
            })
            index_mapping[display_index] = idx
            display_index += 1
        # Gap token
        tokens.append({
            'display_index': display_index,
            'actual_index': None,
            'token_id': None,
            'decoded_token': '... [ Tokens Hidden ] ...',
            'is_generated': False,
            'is_gap': True
        })
        display_index += 1
        # Last 1000 tokens
        for idx in range(total_tokens - 1000, total_tokens):
            info = generated_info[idx]
            tokens.append({
                'display_index': display_index,
                'actual_index': idx,
                'token_id': info['token_id'],
                'decoded_token': info['display_token'],
                'is_generated': info['is_generated']
            })
            index_mapping[display_index] = idx
            display_index += 1

    return render_template('index.html', tokens=tokens, num_layers=num_layers, num_heads=num_heads)

@app.route('/get_token_info', methods=['POST'])
def get_token_info():
    token_actual_index = int(request.form.get('token_actual_index', -1))
    selected_layer = int(request.form.get('selected_layer', 0))
    selected_head = int(request.form.get('selected_head', 0))

    if token_actual_index < 0 or token_actual_index >= len(generated_info):
        return jsonify({'error': 'Invalid token index.'}), 400

    token_info = generated_info[token_actual_index]
    total_tokens = len(generated_info)

    is_initial_context = token_info.get('is_initial_context', False)

    # Safely access 'attn_info_per_layer'
    attn_info_per_layer = token_info.get('attn_info_per_layer', [])
    if selected_layer >= len(attn_info_per_layer):
        return jsonify({'error': 'Selected layer out of range.'}), 400

    attn_info_layer = attn_info_per_layer[selected_layer]

    # Prepare attention arrays
    attention_from_selected = [0.0] * total_tokens  # Attended to by selected token
    attention_to_selected = [0.0] * total_tokens    # Attending to selected token

    # Initialize total_attention_per_token if not present
    total_attention_per_token = [0.0] * total_tokens
    for idx, info in enumerate(generated_info):
        layer_attentions = info.get('total_attention_per_layer_head', [])
        if selected_layer < len(layer_attentions):
            head_attentions = layer_attentions[selected_layer]
            if selected_head < len(head_attentions):
                total_attention_per_token[idx] = float(head_attentions[selected_head])
        else:
            total_attention_per_token[idx] = 0.0

    # Extract data for the selected head
    topk_indices_to = []
    topk_values_to = []
    if 'topk_indices_to' in attn_info_layer and selected_head < len(attn_info_layer['topk_indices_to']):
        topk_indices_to = attn_info_layer['topk_indices_to'][selected_head]
        topk_values_to = attn_info_layer['topk_values_to'][selected_head]

    # Prepare attention_from_selected array for highlighting (Attended To)
    for idx_to, val in zip(topk_indices_to, topk_values_to):
        idx_to = int(idx_to)
        if 0 <= idx_to < total_tokens:
            attention_from_selected[idx_to] = float(val)

    # Get top tokens attended to by this token
    topk_tokens_to = []
    topk_scores_to = []
    topk_distances_to = []
    topk_k_norms_to = []
    topk_v_norms_to = []

    for idx_to, val in zip(topk_indices_to, topk_values_to):
        idx_to = int(idx_to)
        if 0 <= idx_to < total_tokens and val > 0:
            context_tokens = get_context_tokens(idx_to, token_actual_index, total_tokens)
            k_norm = 0.0
            v_norm = 0.0
            # Safely access k_norm and v_norm
            idx_to_token_info = generated_info[idx_to]
            idx_to_token_norms = idx_to_token_info.get('token_norms', {})
            if idx_to_token_norms:
                k_norms = idx_to_token_norms.get('k_norms', [])
                v_norms = idx_to_token_norms.get('v_norms', [])
                if selected_layer < len(k_norms) and selected_head < len(k_norms[selected_layer]):
                    k_norm_value = k_norms[selected_layer][selected_head]
                    k_norm = float(k_norm_value) if isinstance(k_norm_value, torch.Tensor) else k_norm_value
                if selected_layer < len(v_norms) and selected_head < len(v_norms[selected_layer]):
                    v_norm_value = v_norms[selected_layer][selected_head]
                    v_norm = float(v_norm_value) if isinstance(v_norm_value, torch.Tensor) else v_norm_value
            topk_tokens_to.append({
                'token_index': idx_to,
                'decoded_token': idx_to_token_info.get('display_token', ''),
                'context_tokens': context_tokens
            })
            topk_scores_to.append(float(val))
            distance = idx_to - token_actual_index
            topk_distances_to.append(distance)
            topk_k_norms_to.append(k_norm)
            topk_v_norms_to.append(v_norm)

    # Get top tokens attending to this token
    topk_tokens_from = []
    topk_scores_from = []
    topk_distances_from = []
    topk_q_norms_from = []

    attention_scores = token_info.get('attention_scores', {})
    top_tokens_attending_to_list = attention_scores.get('top_tokens_attending_to', [])
    if selected_layer < len(top_tokens_attending_to_list):
        top_tokens_attending_to_heads = top_tokens_attending_to_list[selected_layer]
        if selected_head < len(top_tokens_attending_to_heads):
            top_tokens_attending_to = top_tokens_attending_to_heads[selected_head]
            for idx_from, score in top_tokens_attending_to:
                idx_from = int(idx_from)
                if 0 <= idx_from < total_tokens:
                    attention_to_selected[idx_from] = float(score)
                    context_tokens = get_context_tokens(idx_from, token_actual_index, total_tokens)
                    idx_from_token_info = generated_info[idx_from]
                    idx_from_token_norms = idx_from_token_info.get('token_norms', {})
                    q_norm = 0.0
                    if idx_from_token_norms:
                        q_norms = idx_from_token_norms.get('q_norms', [])
                        if selected_layer < len(q_norms) and selected_head < len(q_norms[selected_layer]):
                            q_norm_value = q_norms[selected_layer][selected_head]
                            q_norm = float(q_norm_value) if isinstance(q_norm_value, torch.Tensor) else q_norm_value
                    topk_tokens_from.append({
                        'token_index': idx_from,
                        'decoded_token': idx_from_token_info.get('display_token', ''),
                        'context_tokens': context_tokens
                    })
                    topk_scores_from.append(float(score))
                    distance = idx_from - token_actual_index
                    topk_distances_from.append(distance)
                    topk_q_norms_from.append(q_norm)

    # Get norms for selected token
    selected_embedding_norm = 0.0
    selected_q_norm = 0.0
    selected_k_norm = 0.0
    selected_v_norm = 0.0
    weighted_v_norm = 0.0
    weighted_v_excl_topk_norm = 0.0

    if attn_info_layer:
        embedding_norms = attn_info_layer.get('embedding_norms', [])
        if selected_head < len(embedding_norms):
            value = embedding_norms[selected_head]
            selected_embedding_norm = float(value) if isinstance(value, torch.Tensor) else value
        q_norms = attn_info_layer.get('q_norms', [])
        if selected_head < len(q_norms):
            value = q_norms[selected_head]
            selected_q_norm = float(value) if isinstance(value, torch.Tensor) else value
        k_norms = attn_info_layer.get('k_norms', [])
        if selected_head < len(k_norms):
            value = k_norms[selected_head]
            selected_k_norm = float(value) if isinstance(value, torch.Tensor) else value
        v_norms = attn_info_layer.get('v_norms', [])
        if selected_head < len(v_norms):
            value = v_norms[selected_head]
            selected_v_norm = float(value) if isinstance(value, torch.Tensor) else value
        weighted_v_norms = attn_info_layer.get('weighted_v_norms', [])
        if selected_head < len(weighted_v_norms):
            value = weighted_v_norms[selected_head]
            weighted_v_norm = float(value) if isinstance(value, torch.Tensor) else value
        weighted_v_excl_topk_norms = attn_info_layer.get('weighted_v_excl_topk_norms', [])
        if selected_head < len(weighted_v_excl_topk_norms):
            value = weighted_v_excl_topk_norms[selected_head]
            weighted_v_excl_topk_norm = float(value) if isinstance(value, torch.Tensor) else value

    # Total attention falling on selected token
    total_attention = 0.0
    if selected_layer < len(token_info['total_attention_per_layer_head']):
        if selected_head < len(token_info['total_attention_per_layer_head'][selected_layer]):
            total_attention_value = token_info['total_attention_per_layer_head'][selected_layer][selected_head]
            total_attention = float(total_attention_value) if isinstance(total_attention_value, torch.Tensor) else total_attention_value

    # Prepare attention arrays to be JSON serializable
    attention_from_selected = [float(val) for val in attention_from_selected]
    attention_to_selected = [float(val) for val in attention_to_selected]
    total_attention_per_token = [float(val) for val in total_attention_per_token]

    # Ensure that next_token_probs are serializable
    next_token_probs_per_layer = token_info.get('next_token_probs_per_layer', [])
    next_token_probs = []
    if selected_layer < len(next_token_probs_per_layer):
        next_token_probs_layer = next_token_probs_per_layer[selected_layer]
        for prob_info in next_token_probs_layer:
            next_token_probs.append({
                'decoded_token': prob_info.get('decoded_token', ''),
                'probability': float(prob_info.get('probability', 0.0))
            })

    response = {
        'is_initial_context': is_initial_context,
        'selected_token_text': token_info['display_token'],
        'selected_token_context': get_context_tokens(token_actual_index, token_actual_index, total_tokens),
        'topk_tokens_to': topk_tokens_to,
        'topk_scores_to': topk_scores_to,
        'topk_distances_to': topk_distances_to,
        'topk_k_norms_to': topk_k_norms_to,
        'topk_v_norms_to': topk_v_norms_to,
        'topk_tokens_from': topk_tokens_from,
        'topk_scores_from': topk_scores_from,
        'topk_distances_from': topk_distances_from,
        'topk_q_norms_from': topk_q_norms_from,
        'attention_from_selected': attention_from_selected,
        'attention_to_selected': attention_to_selected,
        'selected_embedding_norm': selected_embedding_norm,
        'selected_q_norm': selected_q_norm,
        'selected_k_norm': selected_k_norm,
        'selected_v_norm': selected_v_norm,
        'weighted_v_norm': weighted_v_norm,
        'weighted_v_excl_topk_norm': weighted_v_excl_topk_norm,
        'total_attention_on_selected': total_attention,
        'total_attention_per_token': total_attention_per_token,
        'next_token_probs': next_token_probs  # Include next token probabilities with decoded tokens
    }
    return jsonify(response)

def get_context_tokens(idx, selected_idx, total_tokens):
    context_tokens = []
    context_size = 5  # Number of tokens on each side
    start = idx - context_size
    end = idx + context_size + 1
    for idx_ctxt in range(start, end):
        if 0 <= idx_ctxt < total_tokens:
            token_text = generated_info[idx_ctxt].get('display_token', ' ')
            is_center = (idx_ctxt == idx)
            is_selected = (idx_ctxt == selected_idx)
            context_tokens.append({'token': token_text, 'is_center': is_center, 'is_selected': is_selected})
        else:
            # Handle out-of-range indices
            context_tokens.append({'token': ' ', 'is_center': False, 'is_selected': False})
    return context_tokens

def main():
    parser = argparse.ArgumentParser(description="Flask App for Attention Visualization")
    parser.add_argument('info_file', type=str, help='Path to the generated info file (e.g., test.info)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the Flask app on')
    parser.add_argument('--port', type=int, default=5005, help='Port to run the Flask app on')
    args = parser.parse_args()

    load_generated_info(args.info_file)
    preprocess_generated_info()

    # Run the Flask app with the specified host and port
    app.run(debug=True, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
