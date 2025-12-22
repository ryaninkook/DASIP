import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # dim= 10,000 , x = 0,1,2,3,--100 , 100-10,000
        # dim = 1 , 0 1
    def forward(self, x): # ryan added 0 -1 / 0,1,2,3,--100
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
