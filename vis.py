import matplotlib.pyplot as plt
import numpy as np

def plot_attention_from_info(info, name, index_of_interest):
    T = info[0]['initial_context_length']
    attn_info_per_layer = info[-1]['attn_info_per_layer']
    n_layer = len(attn_info_per_layer)

    # Collect (layer_idx, head_idx, indices, values) for all heads
    all_heads_data = []
    for layer_idx, layer_info in enumerate(attn_info_per_layer):
        layer_topk_indices = layer_info['topk_indices_to'].cpu().numpy()  # (n_head, K)
        layer_topk_values = layer_info['topk_values_to'].cpu().numpy()    # (n_head, K)
        n_head = layer_topk_indices.shape[0]
        for head_idx in range(n_head):
            all_heads_data.append((layer_idx, head_idx, layer_topk_indices[head_idx], layer_topk_values[head_idx]))

    # Define the retrieval range: ±4 around index_of_interest
    retrieval_start_idx = max(0, index_of_interest - 4)
    retrieval_end_idx = min(T, index_of_interest + 5)  # end is exclusive

    # Define the local range: last 256 tokens
    local_start_idx = max(0, T - 256)
    local_end_idx = T

    # Define the sink range: first 128 tokens
    sink_start_idx = 0
    sink_end_idx = min(128, T)

    # Define self range: last token (index T-1)
    self_start_idx = T - 1
    self_end_idx = T

    def select_top_heads_for_range(all_heads_data, start_idx, end_idx, top_k=4):
        # Compute sum_in_range for each head
        head_scores = []
        for (layer_idx, head_idx, indices, values) in all_heads_data:
            mask = (indices >= start_idx) & (indices < end_idx)
            sum_in_range = values[mask].sum()
            head_scores.append((sum_in_range, layer_idx, head_idx))
        head_scores.sort(key=lambda x: x[0], reverse=True)
        chosen_heads = head_scores[:top_k]  # top heads
        return chosen_heads

    def select_top_heads_multi_peak(all_heads_data, top_k=4):
        # For each head, find the second highest attention value
        # and also ensure at least one top token is outside the sink and local ranges.
        head_scores = []
        for (layer_idx, head_idx, indices, values) in all_heads_data:
            if len(values) == 0:
                continue
            sorted_values_desc = np.sort(values)[::-1]  # Descending
            if len(sorted_values_desc) < 2:
                # If only one token or none, second highest is first or 0
                second_highest = sorted_values_desc[0] if len(sorted_values_desc) == 1 else 0.0
            else:
                second_highest = sorted_values_desc[1]

            # Check if at least one top token falls outside sink and local ranges
            outside_range_mask = (indices >= sink_end_idx) & (indices < local_start_idx)
            if not np.any(outside_range_mask):
                continue

            head_scores.append((second_highest, layer_idx, head_idx))

        head_scores.sort(key=lambda x: x[0], reverse=True)
        chosen_heads = head_scores[:top_k]
        return chosen_heads

    # Select heads for each category
    chosen_heads_retrieval = select_top_heads_for_range(all_heads_data, retrieval_start_idx, retrieval_end_idx, top_k=4)
    chosen_heads_local = select_top_heads_for_range(all_heads_data, local_start_idx, local_end_idx, top_k=4)
    chosen_heads_sink = select_top_heads_for_range(all_heads_data, sink_start_idx, sink_end_idx, top_k=4)
    chosen_heads_self = select_top_heads_for_range(all_heads_data, self_start_idx, self_end_idx, top_k=4)
    chosen_heads_multipeak = select_top_heads_multi_peak(all_heads_data, top_k=4)

    def bin_attention(indices, values, T, bin_size=10):
        # Bin tokens into groups of size bin_size
        n_bins = (T + bin_size - 1) // bin_size  # ceiling division
        binned_values = np.zeros(n_bins)
        for i, idx_token in enumerate(indices):
            bin_idx = idx_token // bin_size
            if values[i] > binned_values[bin_idx]:
                binned_values[bin_idx] = values[i]
        bin_positions = np.arange(n_bins) * bin_size
        return bin_positions, binned_values

    def plot_selected_heads(name_suffix, chosen_heads, title_suffix,
                            start_line=None, end_line=None,
                            bin_size=10, bar_width=15):
        n_chosen = len(chosen_heads)
        if n_chosen == 0:
            # No heads selected, just return
            return

        # Always make a 2x2 grid for up to 4 heads
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        fig.suptitle(f'{name} {title_suffix}', fontsize=8)

        # Flatten axs for easy indexing
        axs = axs.flatten()

        # Plot each chosen head
        for plot_idx, (score, layer_idx, head_idx) in enumerate(chosen_heads):
            if plot_idx >= len(axs):
                break
            ax = axs[plot_idx]

            # Retrieve original indices and values
            for (lyr, hd, indices, values) in all_heads_data:
                if lyr == layer_idx and hd == head_idx:
                    break

            # Sort by index
            sorted_order = np.argsort(indices)
            sorted_indices = indices[sorted_order]
            sorted_values = values[sorted_order]

            # Bin the data for wider bars
            bin_positions, binned_values = bin_attention(sorted_indices, sorted_values, T, bin_size=bin_size)

            ax.bar(bin_positions, binned_values, width=bar_width, color='blue', align='edge')
            ax.set_title(f'Layer {layer_idx}, Head {head_idx} (Score={score:.4f})')
            ax.set_xlabel('Token Index')
            ax.set_ylabel('Attention Weight')
            ax.grid(True)
            ax.set_xlim(0, T)

            # Add vertical lines if range is specified
            if start_line is not None and end_line is not None:
                ax.axvline(start_line, color='red', linestyle='--')
                ax.axvline(end_line, color='red', linestyle='--')

        # If fewer than 4 heads, remaining subplots remain empty
        for empty_idx in range(n_chosen, 4):
            axs[empty_idx].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{name}{name_suffix}.png")
        plt.close(fig)

    # Plot retrieval figure with two vertical lines
    plot_selected_heads("_retrieval", chosen_heads_retrieval, "(Retrieval Range)",
                        retrieval_start_idx - 20, retrieval_end_idx + 20)

    # Plot local figure with two vertical lines
    plot_selected_heads("_local", chosen_heads_local, "(Local Range)",
                        local_start_idx - 20, local_end_idx + 20)

    # Plot sink figure with lines at 0 and 128
    plot_selected_heads("_sink", chosen_heads_sink, "(Sink Range)",
                        sink_start_idx, sink_end_idx)

    # Plot self figure with lines around the last token (T-1)
    plot_selected_heads("_self", chosen_heads_self, "(Self Range)",
                        self_start_idx - 0.5, self_end_idx - 0.5)

    # Plot multi-peak figure (no lines)
    plot_selected_heads("_multipeak", chosen_heads_multipeak, "(Multi-Peak)")
