"""
Full definition of a GPT Language Model with nGPT option integrated.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from rotary_position_embedding import RotaryEmbedding, apply_rotary_pos_emb
from xpos2_position_embedding import Xpos2Embedding, apply_xpos2_emb
from alibi_relative_position_embedding import build_slopes
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
import logging

from tqdm import tqdm
import heapq

from typing import Union, List

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=self.config.dtype)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=self.config.dtype)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dropout = config.head_dropout
        self.local_heads_during_training = config.local_heads_during_training
        self.local_window_size = config.local_window_size
        self.local_heads_random = config.local_heads_random
        self.top_p = config.top_p
        self.min_p = config.min_p
        self.top_a = config.top_a
        self.silu_before_attn_softmax = config.silu_before_attn_softmax
        self.score_threshold = config.score_threshold
        self.score_scale = config.score_scale
        self.layer_id = layer_id

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5

        if self.config.softmax_scale is None:
            if (self.config.use_nGPT == 0):
                self.softmax_scale = 1.0 / sqrt_head_dim
            if (self.config.use_nGPT == 1):
                self.softmax_scale = sqrt_head_dim
        else:
            self.softmax_scale = self.config.softmax_scale

        print("softmax_scale: ", self.softmax_scale)

        self.alibi_slopes = None
        head_size = self.n_embd // self.n_head

        if config.pe == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(head_size,
                rotary_base=config.rope_base,
                rotary_percentage = config.rope_percentage,
            )
        elif config.pe == 'xpos2':
            max_xpos2_pos = config.block_size * 10
            self.rotary_pos_emb = Xpos2Embedding(
                head_size, rotary_base=config.rope_base,
                max_pos=max_xpos2_pos, decay_base=config.xpos2_decay_base,
                decay_angle=config.xpos2_decay_angle,
                precision=config.precision, adaptive=config.xpos2_adaptive
            )
        elif config.pe == 'alibi':
            self.alibi_slopes = build_slopes(
                num_attention_heads=config.n_head,
                num_attention_heads_alibi=config.n_head,
            ).squeeze().float()

        if config.flash:
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            self.flash = False

        #if not self.flash and not self.use_pseudo_flash:
        #    self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)

        if config.use_nGPT == 1:
            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(config.n_embd, dtype=torch.float32))

        if self.config.softmax_like == 'pre_softmax_soft_threshold':
            #initialize learnable threshold and steepness per head
            self.thr_c = nn.Parameter(2.0 * torch.ones(self.n_head, dtype=torch.float32), requires_grad=True)
            self.stp = nn.Parameter(10.0 * torch.ones(self.n_head, dtype=torch.float32), requires_grad=False)

            print("thr_c:", self.thr_c)
            print("stp:", self.stp)

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x, pos=None, collect_info=False, kv_cache=None, return_kv_cache=False):
        B, _, C = x.size()
        device = x.device

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        #print("kv_cache", kv_cache)

        if kv_cache is not None:
            k0, v0 = kv_cache
            k = torch.cat([k0, k], dim=2)
            v = torch.cat([v0, v], dim=2)
            #shape: (B, n_head, T, C // n_head)

            #print("k0 shape:", k0.shape)

        if return_kv_cache:
            kv_cache = (k, v)
            #need to do now before rotations


        if self.training and self.head_dropout > 0:
            head_mask = torch.ones(self.n_head, device=q.device, dtype=q.dtype)
            drop_indices = torch.randperm(self.n_head)[:self.head_dropout]
            head_mask[drop_indices] = 0
            head_mask = head_mask.bool().view(1, self.n_head, 1, 1)
            q = q.masked_fill(~head_mask, 0.0).clone().detach() * (~head_mask) + q * head_mask
            k = k.masked_fill(~head_mask, 0.0).clone().detach() * (~head_mask) + k * head_mask
            v = v.masked_fill(~head_mask, 0.0).clone().detach() * (~head_mask) + v * head_mask

        if self.training and self.config.scaling_target_sequence_length is not None:
            a = float(self.config.block_size)
            b = float(self.config.scaling_target_sequence_length)
            T = q.size(2)
            if T > 1:
                i = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0)
                scaling_factor = 1 + ((a / b) - 1) * (i / (T - 1))
                scaling_factor = scaling_factor.view(1, 1, T, 1)
            else:
                scaling_factor = torch.tensor(1.0, device=q.device, dtype=q.dtype).view(1, 1, 1, 1)
            q = q * scaling_factor
        elif self.config.softmax_log_k > 0:
            if T > 1:
                _k = float(self.config.softmax_log_k)
                i = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0)
                i[0] = 1
                scaling_factor = (1 - _k + _k * torch.log(i)).to(q.dtype)
                scaling_factor = scaling_factor.view(1, 1, T, 1)
            else:
                scaling_factor = torch.tensor(1.0, device=q.device, dtype=q.dtype).view(1, 1, 1, 1)
            q = q * scaling_factor

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, self.n_head, 1, C // self.n_head)
            q = sqk * self.justnorm(q)
            k = sqk * self.justnorm(k)

            #print(q.shape, k.shape)
            #print(q.norm(dim=-1).mean(), k.norm(dim=-1).mean())
            #print(q.norm(dim=-1).std(), k.norm(dim=-1).std())
            #what are the norms really

        if pos is None or kv_cache is not None:
            pos = torch.arange(0, k.shape[2], dtype=torch.long, device=device)

        if self.config.q_constant_scale != 1.0:
            q = q * self.config.q_constant_scale

        if self.flash:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        #print(q.shape, k.shape)

        if self.flash and not self.config.use_pseudo_flash:

            y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                dropout_p=self.dropout if self.training else 0, softmax_scale=self.softmax_scale, causal=True,
                                window_size=(-1, -1), alibi_slopes=self.alibi_slopes, deterministic=False)
            weighted_v = y.transpose(1, 2)
        elif self.config.use_pseudo_flash and q.shape[2] > self.config.pseudo_flash_chunk_size:
            # Use pseudo-flash attention
            weighted_v, extra_info = self.pseudo_flash_attention(q, k, v, pos, x, collect_info)
        else:
            if self.config.self_extend and k.shape[2] > self.config.pretraining_seq_length and self.config.pe == 'rope':
                # Compute group size dynamically
                denominator = max((self.config.pretraining_seq_length - self.config.window_size), 1)
                g_size = (k.shape[2] - self.config.window_size + denominator - 1) // denominator * 32
                g_size = max(g_size, 1)


                #logging.info(f"g_size: {g_size}")

                w_size = self.config.window_size

                # Compute grouped positions
                g_pos = pos // g_size
                # Compute shift
                shift = w_size - (w_size // g_size)
                # Compute shifted grouped positions
                s_g_pos = g_pos + shift

                # Apply positional encodings for normal attention
                if self.config.pe == 'rope':
                    angles_ngb = self.rotary_pos_emb(pos)  # Shape: (T, hs)
                    ngb_q = apply_rotary_pos_emb(q, angles_ngb)
                    ngb_k = apply_rotary_pos_emb(k, angles_ngb)
                elif self.config.pe == 'xpos2':
                    # Implement xpos2 positional encodings if needed
                    pass

                ngb_k = self.sparsify_k_g(ngb_k)

                # Compute normal attention
                ngb_attn = torch.matmul(ngb_q, ngb_k.transpose(-2, -1)) * self.softmax_scale
                # * (1.0 / math.sqrt(k.size(-1)))
                #ngb_attn = ngb_attn.masked_fill(
                #    torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
                ngb_attn = self.apply_right_aligned_causal_mask(ngb_attn)

                # Apply positional encodings for grouped attention
                if self.config.pe == 'rope':
                    angles_q_grp = self.rotary_pos_emb(s_g_pos)
                    angles_k_grp = self.rotary_pos_emb(g_pos)
                    g_q = apply_rotary_pos_emb(q, angles_q_grp)
                    g_k = apply_rotary_pos_emb(k, angles_k_grp)
                elif self.config.pe == 'xpos2':
                    # Implement xpos2 positional encodings if needed
                    pass

                # Compute grouped attention
                g_attn = torch.matmul(g_q, g_k.transpose(-2, -1)) * self.softmax_scale# * (1.0 / math.sqrt(k.size(-1)))
                #g_attn = g_attn.masked_fill(torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) == 0,
                #                            float('-inf'))
                g_attn = self.apply_right_aligned_causal_mask(g_attn)

                merge_mask = self._compute_merge_mask_chunk(k.shape[2] - q.shape[2], k.shape[2], w_size, device)
                #causal_mask = self._compute_causal_mask_chunk(k.shape[2] - q.shape[2], k.shape[2], device)

                #print(merge_mask)
                #print(merge_mask.sum(dim=-1))
                #print(merge_mask.shape)


                attn = torch.where(merge_mask.unsqueeze(0).unsqueeze(0), ngb_attn, g_attn)
                #attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                """# Create masks
                g_mask = torch.tril(torch.ones(T - w_size, T - w_size, device=device))
                mask = torch.ones(T, T, device=device)
                mask[w_size:, :-w_size] -= g_mask

                # Merge attention scores
                mask = mask.bool()
                attn = torch.where(mask.unsqueeze(0).unsqueeze(0), ngb_attn, g_attn)"""
            else:
                if self.config.pe == 'rope':
                    angles = self.rotary_pos_emb(pos)
                    q = apply_rotary_pos_emb(q, angles)
                    k = apply_rotary_pos_emb(k, angles)
                elif self.config.pe == 'xpos2':
                    pass

                attn = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

                """T_q = q.shape[2]
                T_k = k.shape[2]
                i = torch.arange(T_q, device=device).unsqueeze(-1)  # Shape: (T_q, 1)
                j = torch.arange(T_k, device=device).unsqueeze(0)  # Shape: (1, T_k)
                causal_mask = i + (T_k - T_q) >= j  # Shape: (T_q, T_k)
                attn = attn.masked_fill(causal_mask == 0, float('-inf'))"""
                attn = self.apply_right_aligned_causal_mask(attn)

            if self.training and self.local_heads_during_training > 0:
                T = k.shape[2]
                if self.local_heads_random:
                    head_indices = torch.randperm(self.n_head)[:self.local_heads_during_training]
                else:
                    head_indices = torch.arange(self.local_heads_during_training, device=device)
                local_heads_mask = torch.zeros(self.n_head, device=device, dtype=torch.bool)
                local_heads_mask[head_indices] = True

                i = torch.arange(T, device=device).view(-1, 1)
                j = torch.arange(T, device=device).view(1, -1)
                local_mask = (i - j >= self.local_window_size).bool()
                local_mask = local_mask.unsqueeze(0).unsqueeze(0)
                local_heads_mask_expanded = local_heads_mask.view(1, self.n_head, 1, 1)
                attn = attn.masked_fill(local_heads_mask_expanded & local_mask, float('-inf'))

            if self.config.score_scale is not None and self.config.score_scale != 1.0:
                attn = torch.where(attn < self.config.score_threshold, attn * self.config.score_scale, attn)

            if self.config.relu_before_attn_softmax:
                attn = F.relu(attn)
                attn = attn.masked_fill(torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
                if self.config.relu_neg_inf:
                    attn = attn.masked_fill(attn <= 0, float('-inf'))

            if self.config.silu_before_attn_softmax:
                attn = F.silu(attn)
                attn = attn.masked_fill(torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

            attn_probs = self.softmax_like(attn, k.shape[2] - q.shape[2], v)
            #right-alignment

            attn_probs = self._apply_probability_modifications(attn_probs)

            attn_probs = self.attn_dropout(attn_probs)

            q_norms = q.norm(dim=-1)
            k_norms = k.norm(dim=-1)
            v_norms = v.norm(dim=-1)
            embedding_norms = x.norm(dim=-1).unsqueeze(1).expand(-1, self.n_head, -1)

            if collect_info:
                weighted_v = torch.matmul(attn_probs, v)
                weighted_v_norms = weighted_v.norm(dim=-1)

                effective_top_k = min(5, T)
                topk_values, topk_indices = torch.topk(attn_probs, k=effective_top_k, dim=-1)
                att_probs_excl_topk = attn_probs.clone()
                att_probs_excl_topk.scatter_(
                    dim=-1,
                    index=topk_indices,
                    value=0.0
                )
                weighted_v_excl_topk = torch.matmul(att_probs_excl_topk, v)
                weighted_v_excl_topk_norms = weighted_v_excl_topk.norm(dim=-1)

                topk_values_to, topk_indices_to = torch.topk(attn_probs, k=effective_top_k, dim=-1)
                att_probs_T = attn_probs.transpose(-2, -1)
                topk_values_from, topk_indices_from = torch.topk(att_probs_T, k=effective_top_k, dim=-1)

                extra_info = {
                    'q_norms': q_norms,
                    'k_norms': k_norms,
                    'v_norms': v_norms,
                    'embedding_norms': embedding_norms,
                    'weighted_v_norms': weighted_v_norms,
                    'weighted_v_excl_topk_norms': weighted_v_excl_topk_norms,
                    'topk_indices_to': topk_indices_to,
                    'topk_values_to': topk_values_to,
                    'topk_indices_from': topk_indices_from,
                    'topk_values_from': topk_values_from,
                }
            else:
                weighted_v = torch.matmul(attn_probs, v)
                extra_info = None

        y = weighted_v.transpose(1, 2).contiguous().view(B, -1, C)

        y = self.resid_dropout(self.c_proj(y))

        if collect_info:
            return y, extra_info
        else:
            if return_kv_cache:
                #k shape: (B, n_head, T, C // n_head)
                return y, kv_cache
            return y

    def apply_right_aligned_causal_mask(self, attn_scores):
        # Get the sizes of T_q (queries) and T_k (keys)
        T_q = attn_scores.size(-2)
        T_k = attn_scores.size(-1)
        device = attn_scores.device

        # Generate indices for T_q and T_k
        i = torch.arange(T_q, device=device).unsqueeze(-1)  # Shape: (T_q, 1)
        j = torch.arange(T_k, device=device).unsqueeze(0)  # Shape: (1, T_k)

        # Create the right-aligned causal mask
        causal_mask = (i + (T_k - T_q)) >= j  # Shape: (T_q, T_k)

        # Apply the mask to the attention scores
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        return attn_scores

    def pseudo_flash_attention(self, q, k, v, pos, x, collect_info):
        """
        Pseudo-flash attention implementation to limit memory consumption without using flash attention.
        Processes attention in chunks and handles self-extend logic.
        """
        B, n_head, T, head_dim = q.size()
        device = q.device
        weighted_v = torch.zeros_like(q)
        chunk_size = self.config.pseudo_flash_chunk_size

        # Initialize lists for information collection
        if collect_info:
            q_norms_list = []
            k_norms_list = []
            v_norms_list = []
            embedding_norms_list = []
            weighted_v_norms_list = []
            topk_values_to_list = []
            topk_indices_to_list = []
            topk_values_from_list = []
            topk_indices_from_list = []

        # Check if self_extend is enabled
        if self.config.self_extend and q.shape[2] > self.config.pretraining_seq_length and self.config.pe == 'rope':
            # Assume self_extend is only used in inference
            # Compute group size dynamically
            denominator = max((self.config.pretraining_seq_length - self.config.window_size), 1)
            g_size = (T - self.config.window_size + denominator - 1) // denominator * 32
            g_size = max(g_size, 1)

            w_size = self.config.window_size

            # Compute grouped positions
            g_pos = pos // g_size
            # Compute shift
            shift = w_size - (w_size // g_size)
            # Compute shifted grouped positions
            s_g_pos = g_pos + shift

            for t_start in tqdm(range(0, T, chunk_size)):
                t_end = min(t_start + chunk_size, T)
                t_chunk = t_end - t_start
                q_chunk = q[:, :, t_start:t_end, :]

                # Prepare k and v up to t_end for causal attention
                k_chunk = k[:, :, :t_end, :]
                v_chunk = v[:, :, :t_end, :]

                if not self.config.flash:

                    # Compute normal attention scores (ngb_attn)
                    if self.config.pe == 'rope':
                        angles_ngb_q = self.rotary_pos_emb(pos[t_start:t_end])
                        q_chunk_ngb = apply_rotary_pos_emb(q_chunk, angles_ngb_q)
                        angles_ngb_k = self.rotary_pos_emb(pos[:t_end])
                        k_chunk_ngb = apply_rotary_pos_emb(k_chunk, angles_ngb_k)
                    elif self.config.pe == 'xpos2':
                        # Implement xpos2 positional encodings if needed
                        pass

                    attn_scores_chunk_ngb = torch.matmul(q_chunk_ngb, k_chunk_ngb.transpose(-2, -1)) * self.softmax_scale
                    # * (
                    #        1.0 / math.sqrt(head_dim))

                    # Compute grouped attention scores (g_attn)
                    if self.config.pe == 'rope':
                        angles_grp_q = self.rotary_pos_emb(s_g_pos[t_start:t_end])
                        q_chunk_grp = apply_rotary_pos_emb(q_chunk, angles_grp_q)
                        angles_grp_k = self.rotary_pos_emb(g_pos[:t_end])
                        k_chunk_grp = apply_rotary_pos_emb(k_chunk, angles_grp_k)
                    elif self.config.pe == 'xpos2':
                        # Implement xpos2 positional encodings if needed
                        pass

                    k_chunk_grp = self.sparsify_k_g(k_chunk_grp)

                    attn_scores_chunk_grp = torch.matmul(q_chunk_grp, k_chunk_grp.transpose(-2, -1)) * self.softmax_scale
                    # * (
                    #        1.0 / math.sqrt(head_dim))

                    # Compute the merge mask chunk
                    mask_chunk = self._compute_merge_mask_chunk(t_start, t_end, w_size, device)

                    # Merge attention scores using the mask chunk
                    attn_scores_chunk = torch.where(mask_chunk.unsqueeze(0).unsqueeze(0),
                                                    attn_scores_chunk_ngb,
                                                    attn_scores_chunk_grp)

                    # Compute the causal mask chunk
                    causal_mask_chunk = self._compute_causal_mask_chunk(t_start, t_end, device)

                    # Apply causal mask
                    attn_scores_chunk = attn_scores_chunk.masked_fill(~causal_mask_chunk.unsqueeze(0).unsqueeze(0),
                                                                      float('-inf'))

                    # Apply configurations (score scaling, activations)
                    attn_scores_chunk = self._apply_additional_configs_inference(attn_scores_chunk)

                    # Compute attention probabilities
                    attn_probs_chunk = self.softmax_like(attn_scores_chunk, t_start, v_chunk)

                    # Apply probability modifications (top-k, top-p, etc.)
                    attn_probs_chunk = self._apply_probability_modifications(attn_probs_chunk)
                else:
                    # Apply positional encodings
                    if self.config.pe == 'rope':
                        # Neighbor attention positional embeddings
                        angles_q_neighbor = self.rotary_pos_emb(pos[t_start:t_end])
                        q_chunk_neighbor = apply_rotary_pos_emb(q_chunk, angles_q_neighbor)

                        angles_k_neighbor = self.rotary_pos_emb(pos[:t_end])
                        k_chunk_neighbor = apply_rotary_pos_emb(k_chunk, angles_k_neighbor)

                        # Group attention positional embeddings
                        angles_q_group = self.rotary_pos_emb(s_g_pos[t_start:t_end])
                        q_chunk_group = apply_rotary_pos_emb(q_chunk, angles_q_group)

                        angles_k_group = self.rotary_pos_emb(g_pos[:t_end])
                        k_chunk_group = apply_rotary_pos_emb(k_chunk, angles_k_group)
                    elif self.config.pe == 'xpos2':
                        # Implement xpos2 positional encodings if needed
                        pass

                    # Define window sizes
                    window_size_neighbor = [w_size - 1, 0]  # Left window size
                    window_size_group = [-1, -1]  # No window (full attention)

                    # Attention masks
                    causal = True  # Ensure causal masking

                    # Set dropout_p > 0 to get softmax_lse and attention probabilities
                    dropout_p = 0.0001

                    # Call flash_attn_func for neighbor attention
                    _, softmax_lse_neighbor, attn_probs_neighbor = flash_attn_func(
                        q_chunk_neighbor.transpose(1, 2),  # (B, t_chunk, n_head, head_dim)
                        k_chunk_neighbor.transpose(1, 2),
                        v_chunk.transpose(1, 2),
                        dropout_p=dropout_p,
                        causal=causal,
                        window_size=window_size_neighbor,
                        return_attn_probs=True,
                    )

                    # Call flash_attn_func for group attention
                    _, softmax_lse_group, attn_probs_group = flash_attn_func(
                        q_chunk_group.transpose(1, 2),
                        k_chunk_group.transpose(1, 2),
                        v_chunk.transpose(1, 2),
                        dropout_p=dropout_p,
                        causal=causal,
                        window_size=window_size_group,
                        return_attn_probs=True,
                    )

                    # Reshape outputs to (B, n_head, t_chunk, head_dim)
                    #attn_output_neighbor = attn_output_neighbor.permute(0, 2, 1, 3)
                    #attn_output_group = attn_output_group.permute(0, 2, 1, 3)

                    # Use softmax_lse returned by flash_attn_func
                    # Shape of softmax_lse: (B, t_chunk, n_head)
                    # Compute blending factor using lse_gap
                    lse_gap = softmax_lse_group - softmax_lse_neighbor  # (B, t_chunk, n_head)
                    blending_factor = torch.sigmoid(lse_gap).unsqueeze(-1)  # (B, t_chunk, n_head, 1)

                    # Merge attention probabilities
                    # Reshape attn_probs to (B, n_head, t_chunk, t_end)
                    attn_probs_neighbor = attn_probs_neighbor.permute(0, 2, 1, 3)
                    attn_probs_group = attn_probs_group.permute(0, 2, 1, 3)

                    # Apply causal mask to attention probabilities
                    #causal_mask_chunk = torch.tril(torch.ones(t_chunk, t_end, device=device)).unsqueeze(0).unsqueeze(
                    #    0).bool()
                    #attn_probs_neighbor = attn_probs_neighbor.masked_fill(~causal_mask_chunk, 0.0)
                    #attn_probs_group = attn_probs_group.masked_fill(~causal_mask_chunk, 0.0)

                    # Merge attention probabilities using blending factor
                    attn_probs_chunk = attn_probs_neighbor * blending_factor + attn_probs_group * (
                                1 - blending_factor)

                    # Apply attention modifications
                    attn_probs_chunk = self._apply_probability_modifications(attn_probs_chunk)

                    # Apply attention dropout
                    attn_probs_chunk = self.attn_dropout(attn_probs_chunk)

                # Compute weighted values
                weighted_v_chunk = torch.matmul(attn_probs_chunk, v_chunk)
                weighted_v[:, :, t_start:t_end, :] = weighted_v_chunk

                # Collect information if needed
                if collect_info:
                    q_norms_chunk = q_chunk.norm(dim=-1)
                    k_norms_chunk = k_chunk.norm(dim=-1)
                    v_norms_chunk = v_chunk.norm(dim=-1)
                    embedding_norms_chunk = x[:, t_start:t_end, :].norm(dim=-1).unsqueeze(1).expand(-1, self.n_head, -1)
                    weighted_v_norms_chunk = weighted_v_chunk.norm(dim=-1)

                    effective_top_k = min(5, attn_probs_chunk.size(-1))
                    topk_values_to, topk_indices_to = torch.topk(attn_probs_chunk, k=effective_top_k, dim=-1)
                    att_probs_T_chunk = attn_probs_chunk.transpose(-2, -1)
                    topk_values_from, topk_indices_from = torch.topk(att_probs_T_chunk, k=effective_top_k, dim=-1)

                    q_norms_list.append(q_norms_chunk)
                    k_norms_list.append(k_norms_chunk)
                    v_norms_list.append(v_norms_chunk)
                    embedding_norms_list.append(embedding_norms_chunk)
                    weighted_v_norms_list.append(weighted_v_norms_chunk)
                    topk_values_to_list.append(topk_values_to)
                    topk_indices_to_list.append(topk_indices_to)
                    topk_values_from_list.append(topk_values_from)
                    topk_indices_from_list.append(topk_indices_from)

        else:
            # Standard attention with chunking
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                q_chunk = q[:, :, t_start:t_end, :]
                k_chunk = k[:, :, :t_end, :]
                v_chunk = v[:, :, :t_end, :]

                # Apply positional encodings
                if self.config.pe == 'rope':
                    pos_q_chunk = pos[t_start:t_end]
                    angles_q_chunk = self.rotary_pos_emb(pos_q_chunk)
                    q_chunk = apply_rotary_pos_emb(q_chunk, angles_q_chunk)
                    angles_k_chunk = self.rotary_pos_emb(pos[:t_end])
                    k_chunk = apply_rotary_pos_emb(k_chunk, angles_k_chunk)
                elif self.config.pe == 'xpos2':
                    # Implement xpos2 positional encodings if needed
                    pass

                # Compute attention scores
                attn_scores_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.softmax_scale# * (1.0 / math.sqrt(head_dim))

                # Apply causal mask
                causal_mask = torch.tril(torch.ones(t_end, t_end, device=device)).unsqueeze(0).unsqueeze(0)
                causal_mask_chunk = causal_mask[:, :, t_start:t_end, :t_end]
                attn_scores_chunk = attn_scores_chunk.masked_fill(causal_mask_chunk == 0, float('-inf'))

                # Apply additional masks and configurations per chunk (training-specific)
                attn_scores_chunk = self._apply_additional_configs_training(attn_scores_chunk, t_start, t_end)

                # Compute attention probabilities
                #if self.config.sigmoid_attn:
                #    attn_probs_chunk = torch.sigmoid(attn_scores_chunk - torch.log(1e-8 + T))
                #else:

                attn_probs_chunk = self.softmax_like(attn_scores_chunk, t_start, v_chunk)
                #F.softmax(attn_scores_chunk, dim=-1)

                # Apply probability modifications (top-k, top-p, etc.)
                attn_probs_chunk = self._apply_probability_modifications(attn_probs_chunk)

                # Apply attention dropout
                attn_probs_chunk = self.attn_dropout(attn_probs_chunk)

                # Compute weighted values
                weighted_v_chunk = torch.matmul(attn_probs_chunk, v_chunk)
                weighted_v[:, :, t_start:t_end, :] = weighted_v_chunk

                # Collect information if needed
                if collect_info:
                    q_norms_chunk = q_chunk.norm(dim=-1)
                    k_norms_chunk = k_chunk.norm(dim=-1)
                    v_norms_chunk = v_chunk.norm(dim=-1)
                    embedding_norms_chunk = x[:, t_start:t_end, :].norm(dim=-1).unsqueeze(1).expand(-1, self.n_head, -1)
                    weighted_v_norms_chunk = weighted_v_chunk.norm(dim=-1)

                    effective_top_k = min(5, attn_probs_chunk.size(-1))
                    topk_values_to, topk_indices_to = torch.topk(attn_probs_chunk, k=effective_top_k, dim=-1)
                    att_probs_T_chunk = attn_probs_chunk.transpose(-2, -1)
                    topk_values_from, topk_indices_from = torch.topk(att_probs_T_chunk, k=effective_top_k, dim=-1)

                    q_norms_list.append(q_norms_chunk)
                    k_norms_list.append(k_norms_chunk)
                    v_norms_list.append(v_norms_chunk)
                    embedding_norms_list.append(embedding_norms_chunk)
                    weighted_v_norms_list.append(weighted_v_norms_chunk)
                    topk_values_to_list.append(topk_values_to)
                    topk_indices_to_list.append(topk_indices_to)
                    topk_values_from_list.append(topk_values_from)
                    topk_indices_from_list.append(topk_indices_from)

        # Aggregate collected information if needed
        if collect_info:
            extra_info = {
                'q_norms': torch.cat(q_norms_list, dim=-1),
                'k_norms': torch.cat(k_norms_list, dim=-1),
                'v_norms': torch.cat(v_norms_list, dim=-1),
                'embedding_norms': torch.cat(embedding_norms_list, dim=-1),
                'weighted_v_norms': torch.cat(weighted_v_norms_list, dim=-1),
                'topk_indices_to': torch.cat(topk_indices_to_list, dim=-2),
                'topk_values_to': torch.cat(topk_values_to_list, dim=-2),
                'topk_indices_from': torch.cat(topk_indices_from_list, dim=-2),
                'topk_values_from': torch.cat(topk_values_from_list, dim=-2),
            }
        else:
            extra_info = None

        return weighted_v, extra_info

    def softmax_like(self, scores, t_start=0, v=None):

        if self.config.softmax_like == 'sigmoid_bias':
            bias = torch.arange(t_start + 1, t_start + scores.size(-2) + 1, device=scores.device)
            return F.sigmoid(scores - torch.log(bias[None, None, :, None]))
        elif self.config.softmax_like == 'relu_scaled':
            scores = F.relu(scores)
            scale = torch.arange(t_start + 1, t_start + scores.size(-2) + 1, device=scores.device)
            #print(scores)
            #print(scale)
            #print(scores / scale.unsqueeze(0))
            return scores / scale.unsqueeze(0)
        elif self.config.softmax_like == 'relu_sq_scaled':
            scores = F.relu(scores) ** 2
            scale = torch.arange(t_start + 1, t_start + scores.size(-2) + 1, device=scores.device)
            return scores / scale.unsqueeze(0)
        elif self.config.softmax_like == 'silu_scaled':
            scores = F.silu(scores)
            scale = torch.arange(t_start + 1, t_start + scores.size(-2) + 1, device=scores.device)
            return scores / scale.unsqueeze(0)
        elif self.config.softmax_like == 'sparsemax':
            sorted_input, _ = torch.sort(scores, descending=True, dim=-1)
            cumsum_sorted = torch.cumsum(sorted_input, dim=-1)
            num_classes = scores.size(-1)
            range_tensor = torch.arange(1, num_classes + 1, device=scores.device, dtype=scores.dtype)
            threshold = (cumsum_sorted - 1) / range_tensor
            is_valid = sorted_input > threshold
            k = is_valid.sum(dim=-1, keepdim=True)

            tau = (cumsum_sorted.gather(dim=-1, index=k - 1) - 1).squeeze(-1) / k.squeeze(-1)
            tau = tau.unsqueeze(-1)
            output = torch.clamp(scores - tau, min=0)
            return output
        elif self.config.softmax_like == 'simple_thr_clamp':
            thr, _ = scores.max(dim = -1)
            thr *= 0.2
            scores = torch.clamp(scores - thr.unsqueeze(-1), min=0)
            return scores
        elif self.config.softmax_like == 'pre_softmax_threshold':
            thr, _ = scores.max(dim = -1)
            thr = thr - 1.6
            thr = torch.clamp(thr, min=0)
            thr = thr.unsqueeze(-1)
            scores = scores - thr
            scores = torch.where(scores < 0, scores * 100, scores)
            return F.softmax(scores, dim = -1)
        elif self.config.softmax_like == 'pre_softmax_soft_threshold':
            thr, _ = scores.max(dim=-1)
            #use the threshold parameter
            thr = thr - self.thr_c[None, :, None]
            m = torch.sigmoid((scores - thr.unsqueeze(-1)) * self.stp[None, :, None, None])
            #print("scores:", scores)
            #print("log m:", torch.log(m + 1e-8))
            scores = scores + torch.log(m + 1e-8)
            return F.softmax(scores, dim=-1)
        elif self.config.softmax_like == 'min_p_x_vnorm':
            scores = F.softmax(scores, dim=-1)
            v_norm = v.norm(dim=-1).unsqueeze(-2)
            s_x_vnorm = scores * v_norm
            max_s_x_vnorm, _ = torch.max(s_x_vnorm, dim=-1, keepdim=True)
            s_x_vnorm_thr = torch.clamp(max_s_x_vnorm * 0.2, 0)
            scores = torch.where(s_x_vnorm >= s_x_vnorm_thr, scores, 0)
            scores /= scores.sum(dim=-1, keepdim=True) + 1e-8
            return scores
        elif self.config.softmax_like == 'min_s_x_vnorm_softmax':
            v_norm = v.norm(dim=-1).unsqueeze(-2)
            s_x_vnorm = scores * v_norm
            max_s_x_vnorm, _ = torch.max(s_x_vnorm, dim=-1, keepdim=True)
            s_x_vnorm_thr = torch.clamp(max_s_x_vnorm * 0.2, 0)
            s_x_vnorm_mask = s_x_vnorm >= s_x_vnorm_thr
            scores[~s_x_vnorm_mask] = float('-inf')
            return F.softmax(scores, dim=-1)
        elif self.config.softmax_like == 'min_p_x_vnorm_learn_thr':
            return scores
        else:
            return F.softmax(scores, dim=-1)

    def _compute_merge_mask_chunk(self, t_start, t_end, w_size, device):
        """
        Computes the merge mask for a specific chunk without keeping the entire mask in memory.
        """
        #t_chunk = t_end - t_start
        i = torch.arange(t_start, t_end, device=device).unsqueeze(1)  # Shape: [t_chunk, 1]
        j = torch.arange(t_end, device=device).unsqueeze(0)  # Shape: [1, t_end]

        # Conditions based on the original mask logic
        cond1 = i >= w_size
        #cond2 = j < T - w_size
        cond2 = (i - w_size) >= j

        # Compute the mask chunk
        mask_chunk = ~(cond1 & cond2)
        return mask_chunk

    def _compute_causal_mask_chunk(self, t_start, t_end, device):
        """
        Computes the causal mask for a specific chunk without keeping the entire mask in memory.
        """
        t_chunk = t_end - t_start
        i = torch.arange(t_start, t_end, device=device).unsqueeze(1)  # Shape: [t_chunk, 1]
        j = torch.arange(t_end, device=device).unsqueeze(0)  # Shape: [1, t_end]

        # Causal mask condition
        causal_mask_chunk = i >= j
        return causal_mask_chunk

    def _apply_additional_configs_inference(self, attn_scores_chunk):
        """
        Applies configurations to the attention scores during inference.
        """
        # Apply score scaling and activations
        if self.config.score_scale is not None and self.config.score_scale != 1.0:
            attn_scores_chunk = torch.where(attn_scores_chunk < self.config.score_threshold,
                                            attn_scores_chunk * self.config.score_scale,
                                            attn_scores_chunk)

        if self.config.relu_before_attn_softmax:
            attn_scores_chunk = F.relu(attn_scores_chunk)
            if self.config.relu_neg_inf:
                attn_scores_chunk = attn_scores_chunk.masked_fill(attn_scores_chunk <= 0, float('-inf'))

        if self.config.silu_before_attn_softmax:
            attn_scores_chunk = F.silu(attn_scores_chunk)

        return attn_scores_chunk

    def _apply_additional_configs_training(self, attn_scores_chunk, t_start, t_end):
        """
        Applies additional masks and configurations to the attention scores per chunk during training.
        """
        device = attn_scores_chunk.device

        # Apply local heads during training if configured
        if self.training and self.local_heads_during_training > 0:
            if self.local_heads_random:
                head_indices = torch.randperm(self.n_head)[:self.local_heads_during_training]
            else:
                head_indices = torch.arange(self.local_heads_during_training, device=device)
            local_heads_mask = torch.zeros(self.n_head, device=device, dtype=torch.bool)
            local_heads_mask[head_indices] = True

            i = torch.arange(t_start, t_end, device=device).view(-1, 1)
            j = torch.arange(0, t_end, device=device).view(1, -1)
            local_mask = (i - j >= self.local_window_size).bool()
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)
            local_heads_mask_expanded = local_heads_mask.view(1, self.n_head, 1, 1)
            attn_scores_chunk = attn_scores_chunk.masked_fill(local_heads_mask_expanded & local_mask, float('-inf'))

        # Apply score scaling and activations
        if self.config.score_scale is not None and self.config.score_scale != 1.0:
            attn_scores_chunk = torch.where(attn_scores_chunk < self.config.score_threshold,
                                            attn_scores_chunk * self.config.score_scale,
                                            attn_scores_chunk)

        if self.config.relu_before_attn_softmax:
            attn_scores_chunk = F.relu(attn_scores_chunk)
            if self.config.relu_neg_inf:
                attn_scores_chunk = attn_scores_chunk.masked_fill(attn_scores_chunk <= 0, float('-inf'))

        if self.config.silu_before_attn_softmax:
            attn_scores_chunk = F.silu(attn_scores_chunk)

        return attn_scores_chunk

    def _apply_probability_modifications(self, attn_probs_chunk):
        """
        Applies probability modifications like top-k, top-p filtering to the attention probabilities.
        """
        # Apply top-k after softmax
        if self.config.topk_after_attn_softmax > 0:
            top_n_values, top_n_indices = torch.topk(attn_probs_chunk, self.config.topk_after_attn_softmax, dim=-1)
            mask = torch.zeros_like(attn_probs_chunk)
            mask.scatter_(-1, top_n_indices, 1.0)
            attn_probs_chunk = attn_probs_chunk * mask
            attn_probs_sum = attn_probs_chunk.sum(dim=-1, keepdim=True) + 1e-8
            attn_probs_chunk = attn_probs_chunk / attn_probs_sum

        # Apply top-p filtering
        if self.config.top_p > 0:
            sorted_probs, sorted_indices = torch.sort(attn_probs_chunk, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cumulative_mask = cumulative_probs <= self.config.top_p
            cumulative_mask[..., 0] = True
            sorted_probs = sorted_probs * cumulative_mask
            sorted_probs_sum = sorted_probs.sum(dim=-1, keepdim=True) + 1e-8
            sorted_probs = sorted_probs / sorted_probs_sum
            attn_probs_chunk = torch.zeros_like(attn_probs_chunk).scatter_(-1, sorted_indices, sorted_probs)

        # Apply min_p filtering
        if self.config.min_p > 0:
            max_probs, _ = torch.max(attn_probs_chunk, dim=-1, keepdim=True)
            min_threshold = max_probs * self.config.min_p
            min_p_mask = attn_probs_chunk >= min_threshold
            attn_probs_chunk = attn_probs_chunk * min_p_mask
            attn_probs_sum = attn_probs_chunk.sum(dim=-1, keepdim=True) + 1e-8
            attn_probs_chunk = attn_probs_chunk / attn_probs_sum

        # Apply top_a filtering
        if self.config.top_a > 0:
            max_probs, _ = torch.max(attn_probs_chunk, dim=-1, keepdim=True)
            threshold = (max_probs ** 2) * self.config.top_a
            top_a_mask = attn_probs_chunk >= threshold
            attn_probs_chunk = attn_probs_chunk * top_a_mask
            attn_probs_sum = attn_probs_chunk.sum(dim=-1, keepdim=True) + 1e-8
            attn_probs_chunk = attn_probs_chunk / attn_probs_sum

        return attn_probs_chunk

    def sparsify_k_g(self, k_g):
        B, n_head, T, head_dim = k_g.shape
        #print(k_g.shape)
        if self.config.sparse_k_g:
            if self.config.sparse_k_g_type == 'head_layer':
                token_indices = torch.arange(T, device=k_g.device).unsqueeze(0).unsqueeze(0)
                heads = torch.arange(self.n_head, device=k_g.device).unsqueeze(0).unsqueeze(-1)
                mask_indices = token_indices + heads + self.layer_id
                mask = (mask_indices % self.config.sparse_k_g_mod == 0)
                mask = mask.unsqueeze(-1)  # [1, n_head, T, 1]
                mask[:, :, :self.config.sparse_k_g_keepstart, :] = True
                mask = mask.expand(B, n_head, T, head_dim)  # [B, n_head, T, head_dim]
                k_g = k_g * mask
            elif self.config.sparse_k_g_type == 'head':
                token_indices = torch.arange(T, device=k_g.device).unsqueeze(0).unsqueeze(0)
                heads = torch.arange(self.n_head, device=k_g.device).unsqueeze(0).unsqueeze(-1)
                mask_indices = token_indices + heads
                mask = (mask_indices % self.config.sparse_k_g_mod == 0)
                mask = mask.unsqueeze(-1)  # [1, n_head, T, 1]
                mask[:, :, :self.config.sparse_k_g_keepstart, :] = True
                mask = mask.expand(B, n_head, T, head_dim)  # [B, n_head, T, head_dim]
                k_g = k_g * mask
            elif self.config.sparse_k_g_type == 'fixed':
                token_indices = torch.arange(T, device=k_g.device).unsqueeze(0).unsqueeze(0)
                mask_indices = token_indices
                mask = (mask_indices % self.config.sparse_k_g_mod == 0)
                mask = mask.unsqueeze(-1)  # [1, n_head, T, 1]
                mask[:, :, :self.config.sparse_k_g_keepstart, :] = True
                mask = mask.expand(B, n_head, T, head_dim)  # [B, n_head, T, head_dim]
                k_g = k_g * mask
        return k_g

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_nGPT == 1:
            self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=self.config.dtype)
            self.silu = nn.SiLU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=self.config.dtype)
            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = nn.Parameter(self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))
        else:
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=self.config.dtype)
            self.gelu    = nn.GELU()
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=self.config.dtype)
            self.dropout = nn.Dropout(config.dropout)

        if self.config.modded:
            self.c_proj.weight.data.zero_()

    def forward(self, x):
        if self.config.use_nGPT == 1:
            uv = self.c_fc(x)
            suv = (self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd ** 0.5)))
            uv = suv * uv
            u, v = torch.chunk(uv, 2, dim=-1)
            x = u * self.silu(v)
            x = self.c_proj(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pe: str = 'rope'  # positional embeddings: 'abs', 'rope', 'alibi', 'nope', 'xpos2'
    flash: bool = False  # Should we use Flash Attention if available?
    rope_base: int = 10000  # RoPE base
    rope_percentage: float = 1.0  # rotary_percentage
    rope_wavelengths: Union[str, List] = None  # directly pass wavelengths, oveerriding the base
    xpos2_decay_base: float = 2.0  # Decay base
    xpos2_decay_angle: float = math.pi / 2  # Soft max angle
    xpos2_adaptive: bool = True  # Should we change decay angle if there's risk of overflow
    precision: str = 'float16'  # Precision

    scaling_target_sequence_length: int = None  # Target sequence length for scaling during training

    softmax_log_k: float = 0.0  # 1/T = =(1−k)⋅1+k⋅log(x) where T = pre-softmax temp

    use_nGPT: int = 0  # Whether to use nGPT modifications
    base_scale: float = None  # Base scale for nGPT

    relu_before_attn_softmax: bool = False  # Use ReLU instead of softmax for attention scores
    topk_after_attn_softmax: int = 0  # Keep only top-k attention scores

    relu_neg_inf: bool = False # after RELU also set 0 values to -inf

    # New parameters for SelfExtend
    pretraining_seq_length: int = None  # Defaults to 1k if not set
    window_size: int = 128              # Normal attention window
    self_extend: bool = False           # SelfExtend flag

    head_dropout: int = 0

    local_heads_during_training: int = 0
    local_window_size: int = 128
    local_heads_random: bool = False

    top_p: float = 0.0
    min_p: float = 0.0
    top_a: float = 0.0
    silu_before_attn_softmax: bool = False
    score_threshold: float = 0.0
    score_scale: float = 1.0

    use_pseudo_flash: bool = False
    pseudo_flash_chunk_size: int = 1024

    q_constant_scale: float = 1.0

    softmax_like: str = "softmax"

    softmax_scale: float = None

    modded: bool = False
    do_lns: bool = True

    sparse_k_g: bool = False
    sparse_k_g_type: str = 'head'
    sparse_k_g_mod: int = 2
    sparse_k_g_keepstart: int = 1024

    def __post_init__(self):
        if self.base_scale is None:
            self.base_scale = 1.0 / (self.n_embd ** 0.5)
        if self.pretraining_seq_length is None:
            self.pretraining_seq_length = 1024  # Default pretraining sequence length
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.precision]

class Block(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.n_embd, dtype=torch.float32))

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x, pos=None, collect_info=False, kv_cache=None, return_kv_cache=False):
        if collect_info:
            ln1_out = self.ln_1(x)
            attn_out, attn_info = self.attn(ln1_out, pos=pos, collect_info=collect_info)
            if self.config.use_nGPT == 1:
                lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                lr = torch.abs(lr)

                A_norm = self.justnorm(ln1_out)
                B_norm = self.justnorm(attn_out)

                res = A_norm + lr * (B_norm - A_norm)
                x = self.justnorm(res)
            else:
                x = x + attn_out

            ln2_out = self.ln_2(x)
            mlp_out = self.mlp(ln2_out)
            if self.config.use_nGPT == 1:
                lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
                lr = torch.abs(lr)

                A_norm = self.justnorm(ln2_out)
                B_norm = self.justnorm(mlp_out)

                res = A_norm + lr * (B_norm - A_norm)
                x = self.justnorm(res)
            else:
                x = x + mlp_out
            return x, attn_info
        else:
            if self.config.do_lns:
                ln1_out = self.ln_1(x)
            else:
                ln1_out = x
            attn_out = self.attn(ln1_out, pos=pos, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
            if return_kv_cache:
                attn_out, kv_cache = attn_out
            if self.config.use_nGPT == 1:
                lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                lr = torch.abs(lr)

                A_norm = self.justnorm(ln1_out)
                B_norm = self.justnorm(attn_out)

                res = A_norm + lr * (B_norm - A_norm)
                x = self.justnorm(res)
            else:
                x = x + attn_out

            if self.config.do_lns:
                ln2_out = self.ln_2(x)
            else:
                ln2_out = x
            mlp_out = self.mlp(ln2_out)
            if self.config.use_nGPT == 1:
                lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
                lr = torch.abs(lr)

                A_norm = self.justnorm(ln2_out)
                B_norm = self.justnorm(mlp_out)

                res = A_norm + lr * (B_norm - A_norm)
                x = self.justnorm(res)
            else:
                x = x + mlp_out

            if return_kv_cache:
                return x, kv_cache
            return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict()
        self.transformer['wte'] = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer['drop'] = nn.Dropout(config.dropout)
        self.transformer['h'] = nn.ModuleList([Block(config, lid) for lid in range(config.n_layer)])
        self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        assert self.config.pe in {'abs', 'rope', 'alibi', 'nope', 'xpos2'}, f"Invalid value for pe: {self.config.pe}"

        if self.config.pe == 'abs':
            logging.info('Using absolute positional embeddings (wpe)')
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        elif self.config.pe == 'rope':
            logging.info('Using RoPE positional embeddings')
        elif self.config.pe == 'xpos2':
            logging.info('Using XPOS2 positional embeddings')
        elif self.config.pe == 'alibi':
            logging.info('Using ALiBi positional embeddings')
        else:
            logging.info('No positional embeddings used (NoPE)')

        logging.info(f'rope_wavelengths: {self.config.rope_wavelengths}, type: {type(self.config.rope_wavelengths)}')
        if isinstance(self.config.rope_wavelengths, str) and self.config.rope_wavelengths.startswith("["):
            self.config.rope_wavelengths = self.config.rope_wavelengths.strip("[]").split(",")
        if isinstance(config.rope_wavelengths, List):
            num_rope_dims = int(config.n_embd / config.n_head * config.rope_percentage)
            assert len(
                config.rope_wavelengths) == num_rope_dims, f"num wavelengths in {config.rope_wavelengths} must match num rope dims {num_rope_dims}"

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if not self.config.modded:
            # Weight tying
            self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        else:
            self.lm_head.weight.data.zero_()

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        if config.use_nGPT == 1:
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

        # Report number of parameters
        logging.info("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.pe == 'abs' and non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.use_nGPT == 1:
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and hasattr(module.bias, 'data'):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if self.config.use_nGPT == 1:
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, idx, targets=None, collect_info=False, collect_probs_per_layer=False, kv_cache=None, return_kv_cache=False):
        device = idx.device
        b, t = idx.size()
        s = 0
        if kv_cache is not None:
            s = kv_cache[0][0].shape[2]
            t += s
            #in case we need correct abs position for decoding, I guess
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(s, t, dtype=torch.long, device=device)  # Shape: (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # Shape: (b, t, n_embd)
        if self.config.modded:
            tok_emb = self.justnorm(tok_emb)

        if self.config.pe == 'abs':
            pos_emb = self.transformer.wpe(pos)  # Shape: (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)  # Shape: (b, t, n_embd)
        else:
            x = self.transformer.drop(tok_emb)  # Shape: (b, t, n_embd)

        attn_info_per_layer = [] if collect_info else None
        logits_per_layer = [] if collect_probs_per_layer else None

        # create layer-wise kv_cache
        if kv_cache is None:
            kv_cache_per_layer = [None] * self.config.n_layer
        else:
            kv_cache_per_layer = kv_cache

        #print(len(kv_cache_per_layer))
        #print(kv_cache)
        #print(return_kv_cache)

        for layer_idx, block in enumerate(self.transformer.h):
            if collect_info:
                x, attn_info = block(x, pos=pos, collect_info=collect_info)
                attn_info_per_layer.append(attn_info)
            else:
                x = block(x, pos=pos, kv_cache=kv_cache_per_layer[layer_idx], return_kv_cache=return_kv_cache)
                if return_kv_cache:
                    x, kv_cache_per_layer[layer_idx] = x

                    #print(x, kv_cache_per_layer[layer_idx])

            if collect_probs_per_layer:
                x0 = self.transformer.ln_f(x)
                if targets is not None:
                    layer_logits = self.lm_head(x0)
                else:
                    layer_logits = self.lm_head(x0[:, [-1], :])
                logits_per_layer.append(layer_logits)

        if self.config.do_lns:
            x = self.transformer.ln_f(x)  # Shape: (b, t, n_embd)

        if targets is not None:
            logits = self.lm_head(x)  # Shape: (b, t, vocab_size)
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
            if self.config.modded:
                logits = 30 * torch.tanh(logits / 30)
                logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # Shape: (b, 1, vocab_size)
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
            if self.config.modded:
                logits = 30 * torch.tanh(logits / 30)
                logits = logits.float()
            loss = None

        """if collect_info and collect_probs_per_layer:
            return logits, loss, attn_info_per_layer, logits_per_layer, x
        elif collect_info:
            return logits, loss, attn_info_per_layer, x
        elif collect_probs_per_layer:
            return logits, loss, logits_per_layer
        else:"""

        if return_kv_cache:
            return logits, kv_cache_per_layer

        return logits, loss

    def crop_block_size(self, block_size):
        # Model surgery to decrease the block size if necessary
        # e.g., we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.pe == 'abs':
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        logging.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        logging.info("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            logging.info(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logging.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logging.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, collect_info=False,
                 collect_probs_per_layer=False, decode=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        generated_info = [] if collect_info else None
        logits_per_layer_generated = [] if collect_probs_per_layer else None  # Initialize list to store generated logits

        device = idx.device
        batch_size = idx.size(0)
        assert batch_size == 1, "This generate function currently supports batch_size=1 only."

        total_seq_length = idx.size(1) + max_new_tokens
        num_layers = self.config.n_layer
        num_heads = self.config.n_head  # Exclude aggregated head from per-head calculations

        # Initialize attention_scores for each token
        attention_scores = [
            {
                'top_tokens_attending_to': [
                    [{} for _ in range(num_heads)]  # For each layer, list of dicts for each head
                    for _ in range(num_layers)
                ]
            }
            for _ in range(total_seq_length)
        ]

        # Initialize token norms
        token_norms = []

        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        seq_len = idx_cond.size(1)
        initial_context_length = seq_len  # The length of the initial context

        # Collect info for initial context
        """
        if collect_info or collect_probs_per_layer:
            outputs = self(idx_cond, collect_info=collect_info, collect_probs_per_layer=collect_probs_per_layer)
            if collect_info and collect_probs_per_layer:
                logits, _, attn_info_per_layer, logits_per_layer, hidden_states = outputs
            elif collect_info:
                logits, _, attn_info_per_layer, hidden_states = outputs
            elif collect_probs_per_layer:
                logits, _, logits_per_layer = outputs
        else:
            logits, _ = self(idx_cond)

        if collect_probs_per_layer:
            logits_per_layer_generated.extend(logits_per_layer)

        if collect_info:
            # Decode tokens individually
            token_ids = idx_cond[0].tolist()  # List of token IDs in the initial context
            decoded_tokens = []
            for token_id in token_ids:
                decoded_token = decode([token_id]) if decode else None
                decoded_tokens.append(decoded_token)


            # Initialize token_norms
            token_norms = [{'q_norms': [], 'k_norms': [], 'v_norms': []} for _ in range(seq_len)]

            # Collect norms and attention info
            for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                # Collect norms
                q_norms = layer_attn_info['q_norms'][0]  # Shape: (nh, T)
                k_norms = layer_attn_info['k_norms'][0]
                v_norms = layer_attn_info['v_norms'][0]

                for i in range(seq_len):
                    token_norms[i]['q_norms'].append(q_norms[:, i])  # Shape: (nh,)
                    token_norms[i]['k_norms'].append(k_norms[:, i])
                    token_norms[i]['v_norms'].append(v_norms[:, i])

            # Collect token info
            for i in tqdm(range(seq_len)):
                token_id = token_ids[i]
                decoded_token = decoded_tokens[i]
                decoded_token = decoded_token if decoded_token else None

                token_info = {
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'is_initial_context': True,
                    'attn_info_per_layer': [],
                    'most_similar_tokens': [],
                    'next_token_probs_per_layer': [],  # Initialize per-layer next token probabilities
                }

                #most_similar_tokens = [
                #    {'token_id': tid, 'decoded_token': dtok, 'similarity': sim}
                #    for tid, dtok, sim in zip(sim_token_ids, sim_decoded_tokens, sim_token_sims)
                #]
                #token_info['most_similar_tokens'] = most_similar_tokens

                for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                    layer_token_info = {}
                    for key in [
                        'q_norms', 'k_norms', 'v_norms', 'embedding_norms',
                        'weighted_v_norms', 'weighted_v_excl_topk_norms',
                        'topk_indices_to', 'topk_values_to', 'topk_indices_from', 'topk_values_from'
                    ]:
                        tensor = layer_attn_info[key]
                        if tensor is not None:
                            layer_token_info[key] = tensor[0, :, i]  # Shape depends on key
                        else:
                            layer_token_info[key] = None
                    token_info['attn_info_per_layer'].append(layer_token_info)

                    # Initialize top_tokens_attending_to for tokens in the initial context
                    # Collect top K future tokens within initial context that attend to this token
                    topk_indices_from = layer_attn_info['topk_indices_from'][0, :, i]  # Shape: (nh, k)
                    topk_values_from = layer_attn_info['topk_values_from'][0, :, i]  # Shape: (nh, k)

                    #print(topk_values_from)
                    #print(topk_indices_from)

                    for head_idx in range(num_heads):
                        indices = topk_indices_from[head_idx].tolist()
                        values = topk_values_from[head_idx].tolist()
                        top_tokens_dict = {}
                        for idx_from, attn_score in zip(indices, values):
                            idx_from = int(idx_from)
                            if idx_from <= i or idx_from >= seq_len:
                                continue  # Only consider future tokens within initial context
                            if idx_from in top_tokens_dict:
                                top_tokens_dict[idx_from] += attn_score
                            else:
                                top_tokens_dict[idx_from] = attn_score
                        # Keep top K tokens
                        top_k_attn = 5
                        top_k_tokens = heapq.nlargest(top_k_attn, top_tokens_dict.items(), key=lambda x: x[1])
                        attention_scores[i]['top_tokens_attending_to'][layer_idx][head_idx] = top_k_tokens

                generated_info.append(token_info)

        if collect_info and collect_probs_per_layer:
            # Compute 'next_token_probs_per_layer' for the last token of initial context
            last_token_idx = seq_len - 1
            token_info = generated_info[last_token_idx]  # Get the token_info for the last token
            # Collect next token probabilities per layer
            next_token_probs_per_layer = []
            for layer_logits in logits_per_layer:
                #print(layer_logits.shape)
                layer_logits = layer_logits[:, -1, :]  # Shape: (1, vocab_size)
                layer_probs = F.softmax(layer_logits, dim=-1)
                # Extract top 10 next token probabilities
                top_probs, top_indices = torch.topk(layer_probs, k=10, dim=-1)  # Shape: (1, k)
                next_token_probs_layer = []
                for idx_token, prob in zip(top_indices[0], top_probs[0]):
                    token_id = int(idx_token.item())
                    probability = float(prob.item())
                    decoded_token = decode([token_id]) if decode else None
                    next_token_probs_layer.append({
                        'token_id': token_id,
                        'decoded_token': decoded_token,
                        'probability': probability
                    })
                next_token_probs_per_layer.append(next_token_probs_layer)
            token_info['next_token_probs_per_layer'] = next_token_probs_per_layer
"""
        next_token_probs_list = []

        kv_cache = None

        idx_cond = idx

        use_kv_cache = True

        # Start generating new tokens
        for t in tqdm(range(max_new_tokens), desc="Generating tokens"):
            #[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
            """if collect_info or collect_probs_per_layer:
                outputs = self(idx_cond, collect_info=collect_info, collect_probs_per_layer=collect_probs_per_layer)
                if collect_info and collect_probs_per_layer:
                    logits, _, attn_info_per_layer, logits_per_layer, hidden_states = outputs
                elif collect_info:
                    logits, _, attn_info_per_layer, hidden_states = outputs
                elif collect_probs_per_layer:
                    logits, _, logits_per_layer = outputs
            else:"""

            #kv_cache = None

            if use_kv_cache:
                logits, kv_cache = self(idx_cond, kv_cache=kv_cache, return_kv_cache=True)
            else:
                logits, _ = self(idx_cond)

            #if collect_probs_per_layer:
            #    logits_per_layer_generated.extend(logits_per_layer)

            logits = logits[:, -1, :] / temperature  # Shape: (1, vocab_size)

            if collect_info and collect_probs_per_layer:
                # Collect next token probabilities per layer
                next_token_probs_per_layer = []
                for layer_logits in logits_per_layer:
                    layer_logits = layer_logits[:, -1, :]  # Shape: (1, vocab_size)
                    layer_probs = F.softmax(layer_logits, dim=-1)

                    # Extract top 10 next token probabilities
                    top_probs, top_indices = torch.topk(layer_probs, k=10, dim=-1)  # Shape: (1, k)
                    next_token_probs_layer = []
                    for idx_token, prob in zip(top_indices[0], top_probs[0]):
                        token_id = int(idx_token.item())
                        probability = float(prob.item())
                        decoded_token = decode([token_id]) if decode else None
                        next_token_probs_layer.append({
                            'token_id': token_id,
                            'decoded_token': decoded_token,
                            'probability': probability
                        })
                    next_token_probs_per_layer.append(next_token_probs_layer)

            probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

            # Extract top 10 next token probabilities before sampling
            top_probs, top_indices = torch.topk(probs, k=10, dim=-1)  # Shape: (1, k)
            next_token_probs = []
            for idx_token, prob in zip(top_indices[0], top_probs[0]):
                token_id = int(idx_token.item())
                probability = float(prob.item())
                decoded_token = decode([token_id]) if decode else None
                next_token_probs.append({
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'probability': probability
                })

            """for token_id in [807, 42534, 31675]:
                probability = probs[0, token_id].item()
                decoded_token = decode([token_id]) if decode else None
                next_token_probs.append({
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'probability': probability
                })"""

            if top_k is not None:
                current_top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k=current_top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

            #token_id = idx[:, -1].item()
            #if we are looking at token before generated instead

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)

            idx = torch.cat((idx, idx_next), dim=1)

            if use_kv_cache:
                idx_cond = idx_next#only need last token for decoding with kv_cache
            else:
                idx_cond = idx#torch.cat((idx_cond, idx_next), dim=1)

            token_id = idx_next.item()
            decoded_token = decode([token_id]) if decode else None


            if collect_info:
                token_info = {
                    'token_id': token_id,
                    'decoded_token': decoded_token,
                    'is_initial_context': False,
                    'attn_info_per_layer': [],
                    'next_token_probs': next_token_probs,
                    'most_similar_tokens': [],
                    'next_token_probs_per_layer': next_token_probs_per_layer,  # Add per-layer next token probabilities
                }
                current_token_idx = idx.size(1) - 2  # Index of current token
                token_norm = {
                    'q_norms': [],
                    'k_norms': [],
                    'v_norms': []
                }

                if t > 0:
                    # first one will be duplicate of last initial context token, so only add for t > 0
                    for layer_idx, layer_attn_info in enumerate(attn_info_per_layer):
                        layer_token_info = {}
                        for key in [
                            'q_norms', 'k_norms', 'v_norms', 'embedding_norms',
                            'weighted_v_norms', 'weighted_v_excl_topk_norms',
                            'topk_indices_to', 'topk_values_to', 'topk_indices_from', 'topk_values_from'
                        ]:
                            tensor = layer_attn_info[key]
                            if tensor is not None:
                                layer_token_info[key] = tensor[0, :, -1]  # Shape depends on key
                                # here it's -1 because the attention information is the last one
                            else:
                                layer_token_info[key] = None
                        token_info['attn_info_per_layer'].append(layer_token_info)

                        # Store norms
                        token_norm['q_norms'].append(layer_attn_info['q_norms'][0, :, -1])  # Shape: (nh,)
                        token_norm['k_norms'].append(layer_attn_info['k_norms'][0, :, -1])
                        token_norm['v_norms'].append(layer_attn_info['v_norms'][0, :, -1])

                        # Update attention_scores for tokens attended to by the new token
                        topk_indices = layer_attn_info['topk_indices_to'][0, :, -1]  # Shape: (nh, k)
                        topk_values = layer_attn_info['topk_values_to'][0, :, -1]  # Shape: (nh, k)

                        for head_idx in range(num_heads):
                            indices = topk_indices[head_idx].tolist()
                            values = topk_values[head_idx].tolist()
                            for idx_token, attn_score in zip(indices, values):
                                target_idx = int(idx_token)
                                if target_idx >= idx.size(1) - 1:
                                    continue  # Ignore if index is out of range

                                attention_score = attn_score

                                # Update top tokens attending to the target token
                                top_tokens_dict = dict(
                                    attention_scores[target_idx]['top_tokens_attending_to'][layer_idx][head_idx]
                                )

                                if current_token_idx in top_tokens_dict:
                                    top_tokens_dict[current_token_idx] += attention_score
                                else:
                                    top_tokens_dict[current_token_idx] = attention_score

                                # Keep top K tokens
                                top_k_attn = 5
                                top_k_tokens = heapq.nlargest(top_k_attn, top_tokens_dict.items(), key=lambda x: x[1])
                                attention_scores[target_idx]['top_tokens_attending_to'][layer_idx][
                                    head_idx] = top_k_tokens

                if t > 0:
                    token_norms.append(token_norm)
                    generated_info.append(token_info)

            print(decoded_token)
            print(next_token_probs)
            next_token_probs_list.append(next_token_probs)

            #if decoded_token not in [':', '8', '090', '293', ' 8', ' 090', ' 293']:
            #    break

        if collect_info:
            # Include initial context length in the generated_info
            generated_info[0]['initial_context_length'] = initial_context_length
            # Include attention_scores and token_norms in the generated_info
            for idx_info, info in enumerate(generated_info):
                info['attention_scores'] = attention_scores[idx_info]
                info['token_norms'] = token_norms[idx_info]

            # Compute total attention falling on each token per layer and head
            for idx_info, info in enumerate(generated_info):
                total_attention_per_layer_head = []
                for layer_idx in range(num_layers):
                    layer_total_attention = []
                    for head_idx in range(num_heads):
                        total_attention = sum(
                            score for idx_from, score in
                            info['attention_scores']['top_tokens_attending_to'][layer_idx][head_idx]
                        )
                        layer_total_attention.append(total_attention)
                    total_attention_per_layer_head.append(layer_total_attention)
                info['total_attention_per_layer_head'] = total_attention_per_layer_head

            if collect_probs_per_layer:
                return idx, generated_info, logits_per_layer_generated
            else:
                return idx, generated_info
        else:
            if collect_probs_per_layer:
                return idx, logits_per_layer_generated
            else:
                return idx, next_token_probs_list
