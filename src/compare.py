import torch
import torch.nn.functional as F
from src.eff_mha_torch import EfficientMHA


def test_same():
    batch_size = 1
    embed_dim = 72
    chunk_size = 512
    seq_len = 512
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    net = EfficientMHA(embed_dim=embed_dim, num_heads=1, chunk_size=chunk_size)
    # w = torch.cat([net.q_proj.weight, net.k_proj.weight, net.v_proj.weight], dim=0)
    # net.eval()
    net = net.cuda()

    query = torch.rand(batch_size, seq_len, embed_dim, device='cuda')
    key = torch.rand(batch_size, seq_len, embed_dim, device='cuda')
    value = torch.rand(batch_size, seq_len, embed_dim, device='cuda')

    out = net(query, key, value).view(-1)

    q = query @ net.q_proj.weight * (embed_dim ** -0.5)
    k = key @ net.k_proj.weight
    v = value @ net.v_proj.weight

    o = F.softmax(q @ k.movedim(-2, -1), dim=-1) @ v
    o = o @ net.o_proj.weight
    o = o.view(-1)

    assert torch.allclose(out, o, atol=1e-6), f'{out[:10]}\n{o[:10]}\n{(out/o)[:10]}'


if __name__ == '__main__':
    test_same()
