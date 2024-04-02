from gen_names.models.mamba import SSM, MambaBlock, Mamba
import torch

from gen_names.models.true_model import ModelArgs
from gen_names.models.true_model import Mamba as MB

import lightning as L

L.seed_everything(1702)

# net1 = SSM(
#     inner_dim = 4,
#     debug = False
# )

# net = MambaBlock(
#     emb_dim = 512,
#     inner_dim = 256,
#     d_conv = 4,
#     debug = True
# )

net = Mamba(
    vocab_size = 96,
    emb_dim = 256,
    num_layers = 4,
    inner_dim = 512,
    d_conv = 4
)

args = ModelArgs(
    vocab_size = 96,
    d_model = 256,
    n_layer = 4,
    expand = 2,
    d_conv = 4
)
net2 = MB(args)

b = 2
l = 4
d = 4
emb = 256
vocab = 96

# x = torch.rand((b, l, d))
x = torch.LongTensor([
    [12, 3, 16, 8, 1, 15, 20],
    [16, 10, 90, 81, 13, 0, 0]
])

out = net(x)
# print(out.shape)
print(torch.argmax(out, dim=-1))
# print(out)

print("-----")

out_2 = net2(x)
# print(out_2.shape)
print(torch.argmax(out_2, dim=-1))
# print(out_2)