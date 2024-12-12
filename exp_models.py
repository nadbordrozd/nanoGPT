from .soft_arithmetic import differentiable_addition
import torch
import torch.nn as nn
from torch.nn import functional as Fnn
import math

from .model import GPT, Block, LayerNorm, CausalSelfAttention, MLP
# We assume GPT, Block, LayerNorm, CausalSelfAttention, MLP, GPTConfig are already defined elsewhere.

class FBlock(nn.Module):
    """A Block variant that applies F to the representations of two tokens."""
    def __init__(self, config):
        super().__init__()
        # We reuse the original Block components
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Additional projections for using F
        self.to_ohe = nn.Linear(config.n_embd, 40)
        self.from_ohe = nn.Linear(50, config.n_embd)

    def forward(self, x):
        # original block operations
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # Apply F to the first two tokens
        # x shape: (B, T, C)
        num1_embed = x[:, 0, :]  # (B, C)
        num2_embed = x[:, 1, :]  # (B, C)

        num1_ohe = self.to_ohe(num1_embed) # (B, 40)
        num2_ohe = self.to_ohe(num2_embed) # (B, 40)

        sum_ohe = differentiable_addition(num1_ohe, num2_ohe)   # (B, 50)
        sum_embed = self.from_ohe(sum_ohe) # (B, C)

        # Insert result back into the sequence (e.g., at token index 2)
        x[:, 2, :] = sum_embed

        return x


class GPT_1(GPT):
    """
    A subclass of GPT that replaces one of the standard Blocks with FBlock
    to incorporate function F into the forward pass.
    """
    def __init__(self, config):
        super().__init__(config)
     
        self.transformer.h[config.modified_layer] = FBlock(config)
