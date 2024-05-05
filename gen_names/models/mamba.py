import torch
from torch import nn
import torch.nn.functional as F
import einops
import math
from gen_names.config import Params

class Mamba(nn.Module):
    def __init__(self, args: Params) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(args.model.vocab_size, args.model.embedding_dim)

        self.layers = nn.ModuleList([MambaBlock(
            emb_dim = args.model.embedding_dim,
            inner_dim = args.model.inner_dim,
            d_conv = args.model.d_conv
        ) for _ in range(args.model.num_layers)])

        self.rms_norm = RMSNorm(args.model.embedding_dim)
        self.head = nn.Linear(args.model.embedding_dim, args.model.vocab_size)
    
    def forward(self, x):
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.rms_norm(x)
        x = self.head(x)
        return x


class MambaBlock(nn.Module):
    def __init__(
            self,
            emb_dim:int,
            inner_dim:int,
            d_conv:int,
            debug:bool = False
        ) -> None:
        super().__init__()
        self.rms_norm = RMSNorm(emb_dim)

        self.inner_dim = inner_dim
        self.enter_linear_left = nn.Linear(
            in_features = emb_dim, 
            out_features = self.inner_dim
        )
        self.enter_linear_right = nn.Linear(
            in_features = emb_dim, 
            out_features = self.inner_dim
        )

        self.silu = nn.SiLU()
        self.conv = nn.Conv1d(
            in_channels = self.inner_dim, 
            out_channels = self.inner_dim,
            kernel_size = d_conv,
            groups = self.inner_dim,
            # padding = d_conv - 1
        )
        self.ssm = SSM(
            inner_dim = self.inner_dim,
            debug=debug
        )
        self.out_linear = nn.Linear(
            in_features = self.inner_dim,
            out_features = emb_dim
        )

        self.debug = debug
    
    def forward(self, x):
        if self.debug:
            print("-----")
            print("Here is start of MambaBlock module")
        
        (b, l, d) = x.shape

        x_inner = self.rms_norm(x)

        x_left = self.enter_linear_left(x_inner)
        x_right = self.enter_linear_right(x_inner)

        if self.debug:
            print(f"x.shape = {x.shape}")
            print(f"x_inner.shape = {x_inner.shape}")
            print(f"x_left.shape = {x_left.shape}")
            print(f"x_right.shape = {x_right.shape}")
            print("-----")

        x_right = self.silu(x_right)

        x_left = einops.rearrange(x_left, "b l d -> b d l")
        x_left = self.conv(x_left)#[:, :, :l]

        if x_left.shape[2] < l:
            x_left = F.pad(x_left, (0, l - x_left.shape[2]))

        x_left = einops.rearrange(x_left, "b d l -> b l d")
        x_left = self.silu(x_left)
        x_left = self.ssm(x_left)

        if self.debug:
            print(f'x_left.shape = {x_left.shape}')
            print(f'x_right.shape = {x_right.shape}')
            print("-----")

        x_inner = x_left * x_right

        x_inner = self.out_linear(x_inner)

        if self.debug:
            print("Here is the end of MambaBlock module")
            print("-----")

        return x_inner + x


class SSM(nn.Module):
    def __init__(
            self,
            inner_dim:int,
            debug:bool = False
        ) -> None:
        super().__init__()
        
        # Для репрезентации A.shape = (N, N)
        latent_dim = inner_dim
        self.hippo = True

        #-------------------------------------------------------------------------------
        self.dt_rank = math.ceil(inner_dim / 16)
        #-------------------------------------------------------------------------------
        if self.hippo:
            A = self.make_HiPPO(inner_dim)
        else:
            A = einops.repeat(torch.arange(1, latent_dim + 1), 'n -> d n', d = inner_dim)
        self.A_log = nn.Parameter(torch.log(-A))
        self.D = nn.Parameter(torch.ones(inner_dim))

        self.make_B = nn.Linear(inner_dim, latent_dim)
        self.make_C = nn.Linear(inner_dim, latent_dim)
        self.make_delta_intermediate = nn.Linear(inner_dim, self.dt_rank)
        self.make_delta = nn.Linear(self.dt_rank, inner_dim)

        self.n = latent_dim
        self.debug = debug
    
    def make_HiPPO(self, N):
        P = torch.sqrt(1 + 2 * torch.arange(N))
        A = P[:, None] * P[None, :]
        A = torch.tril(A) - torch.diag(torch.arange(N))
        return -A
    
    def forward(self, x):
        if self.debug:
            print("-----")
            print("Here is start of SSM module")
        
        b, l, d = x.shape

        A = -torch.exp(self.A_log.float()) # A.shape = (d, n)
        D = self.D.float() # D.shape = (d)

        #--------------------------------------------

        B = self.make_B(x) # B.shape = (b, l, d)
        C = self.make_C(x) # C.shape = (b, l, d)

        delta = self.make_delta_intermediate(x)
        delta = self.make_delta(delta)
        delta = F.softplus(delta)

        #--------------------------------------------

        if self.debug:
            print(f"A.shape = {A.shape}")
            print(A)
            print(f"B.shape = {B.shape}")
            print(B)
            print(f"C.shape = {C.shape}")
            print(C)
            print(f"D.shape = {D.shape}")
            print(D)
            print(f"delta.shape = {delta.shape}")
            print(delta)
            print("-----")

        # Дискретизация
        deltaA = einops.einsum(delta, A, 'b l d, d n -> b l d n')
        A_hat = torch.exp(deltaA)
        if A.shape[0] == A.shape[1] and self.hippo:
            if self.debug:
                print("Matrix `A` represents structured NxN matrix")
            deltaB = einops.einsum(delta, B, 'b l d, b l n -> b l d n')
            I = torch.eye(self.n).to(A_hat.device)
            B_hat = torch.inverse(deltaA) @ (A_hat - I) *  deltaB
            # Для ускорения и так не быстрого процесса можно `x` сразу домножить, чтобы не делать это последовательно потом
            # В ином случае просто закомментить строку ниже и закомментить/раскомментить строку уже в цикле for внизу
            B_hat = einops.einsum(B_hat, x, "b l d n, b l d -> b l d n")
        else:
            # Упрощённая дискретизация матрицы B, вместе с домножением сразу на `x` для ускорения (объяснение чуть ниже)
            B_hat = einops.einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
            # B_hat = einops.einsum(delta, B, 'b l d, b l n -> b l d n')

        if self.debug:
            print(f"A_hat.shape = {A_hat.shape}")
            print(A_hat)
            print(f"B_hat.shape = {B_hat.shape}")
            print(B_hat)
            print("-----")

        # Матрица K
        k = torch.zeros((b, d, self.n)).to(A_hat.device)
        ys = []

        if self.debug:
            only_one = True

        for i in range(l):
            k = A_hat[:, i, :, :] * k + B_hat[:, i, :, :]

            if self.debug:
                if only_one:
                    print(f"A_hat[:, i, :, :].shape = {A_hat[:, i, :, :].shape}")
                    print(f"B_hat[:, i, :, :].shape = {B_hat[:, i, :, :].shape}")
                    print(f"k.shape = {k.shape}")
                    print(f"C[:, i, :].shape = {C[:, i, :].shape}")
                    print("-----")
                    only_one = False

            y = einops.einsum(k, C[:, i, :], 'b d_in n, b n -> b d_in')
            # y = einops.einsum(C[:, i, :], k, x[:, i, :], 'b n, b d n, b d -> b d')
            ys.append(y)
        
        if self.debug:
            print(f"len(ys) = {len(ys)}")
            print(f"ys[0].shape = {ys[0].shape}")
            print("-----")

        y = torch.stack(ys, dim=1)  # y.shape = (b, l, d)

        if self.debug:
            print("Here is the end of SSM module")
            print("-----")

        return y + (x * D)

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5
        ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output