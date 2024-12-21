from .soft_arithmetic import differentiable_addition
import torch
import torch.nn as nn
from torch.nn import functional as Fnn
import math

from .model import GPT, Block, LayerNorm, CausalSelfAttention, MLP
# We assume GPT, Block, LayerNorm, CausalSelfAttention, MLP, GPTConfig are already defined elsewhere.

class FBlock_V1(nn.Module):
    """A Block variant that applies F to the representations of two tokens."""
    def __init__(self, config):
        super().__init__()
        self.D = config.n_digits

        # We reuse the original Block components
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Additional projections for using F
        self.to_ohe = nn.Linear(config.n_embd, self.D * 10)
        self.from_ohe = nn.Linear((self.D + 1) * 10, config.n_embd)

    def forward(self, x):
        # original block operations
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # Apply F to the first two tokens
        # x shape: (B, T, C)
        num1_embed = x[:, 0, :]  # (B, C)
        num2_embed = x[:, 1, :]  # (B, C)



        num1_ohe = self.to_ohe(num1_embed)             # (B, D * 10)        
        num1_ohe = num1_ohe.view(-1, self.D, 10)       # reshape to (B, D, 10)
        num1_ohe = Fnn.softmax(num1_ohe, dim=-1)       # apply softmax across the last dimension
        num1_ohe = num1_ohe.view(-1, self.D * 10)      # back to (B, D * 10)

        num2_ohe = self.to_ohe(num2_embed)             # (B, D * 10)        
        num2_ohe = num2_ohe.view(-1, self.D, 10)       # reshape to (B, D, 10)
        num2_ohe = Fnn.softmax(num2_ohe, dim=-1)       # apply softmax across the last dimension
        num2_ohe = num2_ohe.view(-1, self.D * 10)      # back to (B, D * 10)

        sum_ohe = differentiable_addition(num1_ohe, num2_ohe, self.D)   # (B, (n_digits + 1 ) * 10)
        sum_ohe = sum_ohe.view(-1, self.D + 1, 10)
        sum_ohe = Fnn.softmax(sum_ohe, dim=-1)
        sum_ohe = sum_ohe.view(-1, (self.D + 1) * 10)
        sum_embed = self.from_ohe(sum_ohe) # (B, C)

        x = torch.cat([x, sum_embed.unsqueeze(1)], dim=1)
        return x


class FBlock_V2(nn.Module):
    """A Block variant that applies F to the representations of two tokens."""
    def __init__(self, config):
        super().__init__()
        self.D = config.n_digits

        # We reuse the original Block components
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Additional projections for using F
        self.to_ohe = nn.Linear(config.n_embd, self.D * 10)
        self.from_ohe = nn.Linear((self.D + 1) * 10, config.n_embd)

    def forward(self, x):
        # original block operations
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # Apply F to the first two tokens
        # x shape: (B, T, C)
        num1_embed = x[:, 0, :]  # (B, C)
        num2_embed = x[:, 1, :]  # (B, C)



        num1_ohe = self.to_ohe(num1_embed)             # (B, D * 10)        
        num1_ohe = num1_ohe.view(-1, self.D, 10)       # reshape to (B, D, 10)
        num1_ohe = Fnn.softmax(num1_ohe, dim=-1)       # apply softmax across the last dimension
        num1_ohe = num1_ohe.view(-1, self.D * 10)      # back to (B, D * 10)

        num2_ohe = self.to_ohe(num2_embed)             # (B, D * 10)        
        num2_ohe = num2_ohe.view(-1, self.D, 10)       # reshape to (B, D, 10)
        num2_ohe = Fnn.softmax(num2_ohe, dim=-1)       # apply softmax across the last dimension
        num2_ohe = num2_ohe.view(-1, self.D * 10)      # back to (B, D * 10)

        sum_ohe = differentiable_addition(num1_ohe, num2_ohe, self.D)   # (B, (n_digits + 1 ) * 10)
        sum_ohe = sum_ohe.view(-1, self.D + 1, 10)
        sum_ohe = Fnn.softmax(sum_ohe, dim=-1)
        sum_ohe = sum_ohe.view(-1, (self.D + 1) * 10)
        sum_embed = self.from_ohe(sum_ohe) # (B, C)


        x = torch.cat([x[:, :-1, :], sum_embed.unsqueeze(1)], dim=1)
        return x


class GPT_V1(GPT):
    """
    A subclass of GPT that replaces one of the standard Blocks with FBlock
    to incorporate function F into the forward pass.
    """
    def __init__(self, config):
        super().__init__(config)
     
        self.transformer.h[config.modified_layer] = FBlock_V1(config)

    def forward(self, idx, targets=None):
        """
        Same logic as GPT.forward, but we slice off any extra tokens
        introduced by the last block before computing loss.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
        
        # Positions 0..t-1
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through the Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Now x might have shape (B, T, C) or (B, T+1, C), depending on FBlock concatenation
        
        # In case FBlock appended tokens, let's slice back to T:
        # (only do this if x is longer than T)
        final_seq_len = x.shape[1]
        if final_seq_len > t:
            x = x[:, :t, :]  # drop extra tokens so shape is (B, T, C)
        
        # Final layer norm
        x = self.transformer.ln_f(x)  # (B, T, C)
        
        # If we have targets, compute cross-entropy over all T tokens
        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = Fnn.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # In inference mode, we only need the logits at the last position
            logits = self.lm_head(x[:, [-1], :])  # shape (B, 1, vocab_size)
            loss = None
        
        return logits, loss

class GPT_V2(GPT):
    """
    A subclass of GPT that replaces one of the standard Blocks with FBlock
    to incorporate function F into the forward pass.
    """
    def __init__(self, config):
        super().__init__(config)
     
        self.transformer.h[config.modified_layer] = FBlock_V2(config)
