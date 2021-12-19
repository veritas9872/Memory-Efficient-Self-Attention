import torch
from torch import nn, Tensor

from src.eff_mha_torch import EfficientMHA


def test_same():
    embed_dim = 72
    chunk_size = 128
    seq_len = 512
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, bias=False, batch_first=True)
    b = EfficientMHA(embed_dim=embed_dim, num_heads=1, chunk_size=chunk_size)
    w = torch.cat([b.q_proj.weight, b.k_proj.weight, b.v_proj.weight], dim=0)
    a.in_proj_weight.set_(w)
    a.out_proj.weight.set_(b.o_proj.weight)
    a.eval()
    b.eval()
    a = a.cuda()
    b = b.cuda()

    q = torch.rand(16, seq_len, embed_dim, device='cuda')
    k = torch.rand(16, seq_len, embed_dim, device='cuda')
    v = torch.rand(16, seq_len, embed_dim, device='cuda')

    aa = a(q, k, v)[0]
    bb = b(q, k, v)

    assert torch.allclose(aa, bb, atol=1e-6), (aa - bb).abs().max()


if __name__ == '__main__':
    test_same()
