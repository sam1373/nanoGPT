import torch
from einops import rearrange
from torch import einsum, nn
from typing import List
import logging

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

def _rotate_half(x):
    """
    change sign so the last dimension
    [A, B, C, D] -> [-C, -D, A, B]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        rotary_base: int = 10000,
        rotary_percentage: float = 1.0,
        wavelengths: List = None,
    ):
        """
        Args:

            dim (int): rotary embedding dimension
            rotary_base (int): rotary_base for the positional frequency (default: 10000)
        """
        super().__init__()
        self.rotary_base = rotary_base
        rotary_dim = int(dim * rotary_percentage)
        logging.info(f'rotary_base: {rotary_base}, rotary_percentage: {rotary_percentage}, dim: {dim}, rotary_dim: {rotary_dim}')
        self.wavelengths = wavelengths
        if wavelengths is not None:
            logging.info(f'using passed in wavelengths {self.wavelengths}')
            wavelengths = torch.tensor(wavelengths, dtype=torch.float, device=torch.cuda.current_device())
            inv_freq = 2 * torch.pi / wavelengths
        else:
            inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        # positions: tensor of shape (T,)
        angles = torch.einsum("i , j -> i j", positions.to(self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((angles, angles), dim=-1)  # Shape: (T, dim)
        return emb

def apply_rotary_pos_emb(t, angles):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """

    # assume right-aligned
    if angles.shape[0] != t.shape[2]:
        # print(f"angles shape {angles.shape} is not equal to t shape {t.shape}")
        angles = angles[-t.shape[2]:]

    rot_dim = angles.shape[-1]
    # if t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * angles.cos()) + (_rotate_half(t) * angles.sin())
    return torch.cat((t, t_pass), dim=-1)

