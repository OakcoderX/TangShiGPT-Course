"""A small GPT that maps a sequence of characters to next-character logits."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    width: int = 64
    layers: int = 2
    heads: int = 4
    context: int = 48
    eps: float = 1e-6

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.context) < 1 or self.width % self.heads:
            raise ValueError("层数和维度必须为正，宽度必须能整除注意力头数")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.norm1 = nn.RMSNorm(width, eps=config.eps)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width, eps=config.eps)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.heads
        packed = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, head_width)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
        # At each position, keys and values can come only from this position or earlier.
        read = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(read.transpose(1, 2).reshape(batch, length, width))
        return x + self.down(F.relu(self.up(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.width, eps=config.eps)
        self.head = nn.Linear(config.width, vocab_size, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
