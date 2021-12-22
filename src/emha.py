"""
My implementation of a memory efficient Multi-Head Attention module.

This version is kept simple by design so as not to confuse users.
Also, this version does not attempt to match traditional MHA exactly.
It uses the sigmoid function by default instead of the softmax.
My opinion is that the softmax unnecessarily complicates the self-attention process
while not providing much benefit, if any, over other activation functions
that do not require the entire row of self-attention.
"""

import math

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_
from torch.utils.checkpoint import checkpoint


class ChunkedMHA(nn.Module):
    """
    A skeleton implementation of Chunked MHA.
    No masking has been implemented to simplify the code.

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
        self.scaling = self.head_dim ** -0.5
        assert embed_dim % num_heads == 0, \
            f'Invalid head number {num_heads} for embedding dimension {embed_dim}.'
        self.chunk_size = chunk_size
        self.att_func = torch.sigmoid

        self.q_proj = nn.Linear(
            in_features=embed_dim,
            out_features=self.k_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            in_features=embed_dim,
            out_features=self.k_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            in_features=embed_dim,
            out_features=self.v_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            in_features=embed_dim,
            out_features=self.k_dim,
            bias=False
        )
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.o_proj.weight)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            output: Tensor = None  # No memory savings for now.
    ) -> Tensor:
        n_q, l_q, e_q = query.shape  # N, L, E
        n_k, s_k, e_k = key.shape    # N, S, E
        n_v, s_v, e_v = value.shape  # N, S, E
        assert n_q == n_k == n_v, f'Inconsistent batch sizes {n_q}, {n_k}, {n_v}.'
        assert s_k == s_v, f'Inconsistent source sequence lengths {s_k}, {s_v}.'
        assert e_q == e_k == e_v == self.embed_dim, \
            f'Inconsistent number of feature dimensions {e_q}, {e_k}, {e_v}.'

        c_s = self.chunk_size
        cond1 = (l_q > c_s) and (l_q % c_s != 0)
        cond2 = (s_k > c_s) and (s_k % c_s != 0)
        if cond1 or cond2:
            raise RuntimeError('Inputs not divisible by chunk_size')

        # `tensor_split` is a view operation. No extra memory is allocated.
        qcs = torch.tensor_split(query, math.ceil(l_q / c_s), dim=1)
        kcs = torch.tensor_split(key, math.ceil(s_k / c_s), dim=1)
        vcs = torch.tensor_split(value, math.ceil(s_v / c_s), dim=1)

        if isinstance(output, Tensor):
            assert output.shape == (n_q, l_q, e_v)
            ocs = torch.tensor_split(output, math.ceil(l_q / c_s), dim=1)
        else:
            ocs = None

        result = list()

        for i, qc in enumerate(qcs):
            buffer: Tensor = 0
            for kc, vc in zip(kcs, vcs):  # Gradient checkpointing.
                buffer += checkpoint(self._att, qc, kc, vc)

            if ocs is not None:
                # This will replace values while passing the gradients, I hope.
                # Whether gradients are passed properly needs confirmation.
                ocs[i].put_(torch.arange(ocs[i].numel()), buffer)
            else:
                result.append(buffer)

        if ocs is None:
            output = torch.cat(result, dim=-2, out=output)
        return output


    def _att(
            self,
            qc: Tensor,
            kc: Tensor,
            vc: Tensor,
    ):
        # Attention without the whole-row softmax.
        q = self.q_proj(qc)
        k = self.k_proj(kc)
        v = self.v_proj(vc)

        n_h = self.num_heads
        h_d = self.head_dim

        n_q, c_l, e_q = qc.shape  # N, C, E
        n_k, c_k, e_k = kc.shape  # N, C, E
        n_v, c_v, e_v = vc.shape  # N, C, E

        # Reshape is used in case input is not contiguous.
        q = q.reshape(n_q, c_l, n_h, h_d).movedim(-3, -2) * self.scaling  # \sqrt(d)
        k = k.reshape(n_k, c_k, n_h, h_d).movedim(-3, -2)
        v = v.reshape(n_v, c_v, n_h, h_d).movedim(-3, -2)
        v = v.movedim(-2, -1).contiguous().movedim(-1, -2)
        o = self.att_func(q @ k.movedim(-2, -1)) @ v
        o = o.movedim(-2, -3).reshape(n_q, c_l, e_v)
        return self.o_proj(o.contiguous())
