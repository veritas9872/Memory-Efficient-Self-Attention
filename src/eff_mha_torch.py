"""
Native PyTorch implementation of efficient self-attention.
This is not a copy of the JAX version but a
PyTorch implementation based on first-principles.

Warning: This version produces incorrect results.
"""
import math

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_
from torch.utils.checkpoint import checkpoint


class EfficientMHA(nn.Module):
    """
    A skeleton implementation of efficient MHA based on
    "Self-attention Does Not Need $O(n^2)$ Memory".
    No masking has been implemented to simplify the code.

    This is a native PyTorch implementation based on
    first-principles, not a JAX translation.
    PyTorch does not have an efficient equivalent of `jax.lax.scan`.
    This version also includes the mini-batch dimension.

    Assumes batch-first ordering of input tensors.
    Also assumes the same number of embedding dimensions for k, q, v.

    Note that for best performance, the axis where
    matrix multiplication occurs should be contiguous.
    """
    def __init__(self, embed_dim: int, num_heads: int, chunk_size: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_dim = embed_dim
        self.v_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, \
            f'Invalid head number {num_heads} for embedding dimension {embed_dim}.'
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=False
        )
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.o_proj.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        n_q, l_q, e_q = query.shape
        n_k, s_k, e_k = key.shape
        n_v, s_v, e_v = value.shape
        assert n_q == n_k == n_v, f'Inconsistent batch sizes {n_q}, {n_k}, {n_v}.'
        assert s_k == s_v, f'Inconsistent source sequence lengths {s_k}, {s_v}.'
        assert e_q == e_k == self.k_dim, \
            f'Inconsistent number of feature dimensions {e_q}, {e_k}.'
        assert e_v == self.v_dim, f'Inconsistent number of feature dimensions {e_v}.'

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        n_h = self.num_heads
        h_d = self.head_dim  # Assumes embedding dimensions are the same for all inputs.
        c_s = self.chunk_size
        cond1 = (l_q > c_s) and (l_q % c_s != 0)
        cond2 = (s_k > c_s) and (s_k % c_s != 0)
        if cond1 or cond2:
            raise RuntimeError('Input not divisible by chunk_size')
        # N, (S|L), E -> N, (S|L), H, D -> N, H, (S|L), D.
        # N: Batch size, S: Source length, L: Target length, E: Embedding dims,
        # H: Number of attention heads, D: Feature dims, where D = E // H.
        q = q.view(n_q, l_q, n_h, h_d).movedim(-3, -2) * (h_d ** -0.5)
        k = k.view(n_k, s_k, n_h, h_d).movedim(-3, -2)
        v = v.view(n_v, s_v, n_h, h_d).movedim(-3, -2)
        # Memory optimization for matrix multiplication
        # by making the sequence dimension contiguous.
        v = v.movedim(-2, -1).contiguous().movedim(-1, -2)
        # Split along sequence lengths. `tensor_split` is a view operation.
        qs = q.tensor_split(math.ceil(l_q / c_s), dim=-2)
        ks = k.tensor_split(math.ceil(s_k / c_s), dim=-2)
        vs = v.tensor_split(math.ceil(s_v / c_s), dim=-2)

        result = list()
        for qc in qs:  # Chunked matrix multiplication.
            num_buff = list()
            den_buff = list()
            max_buff = list()
            for kc, vc in zip(ks, vs):  # Summation on one chunked row of the output.
                # Gradient checkpointing for memory-efficient backprop.
                numerator, denominator, max_score = \
                    checkpoint(self.summarize_chunk, qc, kc, vc)
                num_buff.append(numerator)
                den_buff.append(denominator)
                max_buff.append(max_score)
            # The maximum value of the self-attention rows are now available.
            num_buff = torch.stack(num_buff, dim=-1)
            den_buff = torch.stack(den_buff, dim=-1).unsqueeze(-2)
            max_buff = torch.stack(max_buff, dim=-1).unsqueeze(-2)
            row_max, _ = torch.max(max_buff, dim=-1, keepdim=True)
            max_diffs = torch.exp(max_buff - row_max)
            # Scalar multiplication along the row axis is exchangeable.
            num_buff = torch.sum(num_buff * max_diffs, dim=-1)
            den_buff = torch.sum(den_buff * max_diffs, dim=-1)
            den_buff = den_buff.sum(dim=-1, keepdim=True)
            assert num_buff.shape == (n_q, n_h, c_s, h_d), num_buff.shape
            # C_L: Chunk length of target sequence.
            result.append(num_buff / den_buff)  # N, H, C_L, D
        result = torch.cat(result, dim=-2)  # N, H, L, D
        assert result.shape == (n_q, n_h, l_q, h_d), f'{result.shape}.'

        # Return to N, L, E shape for the output.
        result = result.movedim(-2, -3).reshape(n_q, l_q, e_v).contiguous()
        result = self.o_proj(result)  # Output projection.
        return result

    @staticmethod
    def summarize_chunk(qc: Tensor, kc: Tensor, vc: Tensor):
        att = qc @ kc.movedim(-2, -1)
        max_score, _ = torch.max(att, dim=-1, keepdim=True)
        max_score = max_score.detach()  # Max score should not backprop.
        # Subtract chunk row-wise maximum for numerical stability.
        exp_att = torch.exp(att - max_score)  # N, H, C_L, C_S.
        numerator = exp_att @ vc  # Numerator for softmax. N, H, C_L, D.
        # Sum along rows for softmax denominator.
        denominator = exp_att.sum(dim=-1)  # N, H, C_L.
        return numerator, denominator, max_score.squeeze(-1)
