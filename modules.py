import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.resblock(x)

