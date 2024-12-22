from .soft_arithmetic import differentiable_addition
import torch
import torch.nn as nn
from torch.nn import functional as Fnn
import math

from .model import GPT, Block, LayerNorm, CausalSelfAttention, MLP
# We assume GPT, Block, LayerNorm, CausalSelfAttention, MLP, GPTConfig are already defined elsewhere.

class FBlock_V1(Block):
    """
    A variant of the standard Transformer Block that:
      1) Takes two tokens representing two n-digit numbers,
      2) Converts them to OHE (n_digits * 10),
      3) Runs F to get a (n_digits + 1)-digit sum,
      4) Splits this into (n_digits + 1) 'tokens',
      5) Concatenates those tokens to the output sequence.
    """
    def __init__(self, config):
        super().__init__(config)  # calls the original Block.__init__
        self.n_digits = config.n_digits  # number of digits per input number

        # Projection to/from OHE:
        # to_ohe maps embedding -> n_digits*10
        # from_ohe maps 10 -> embedding, used repeatedly for each digit in the sum
        self.to_ohe = nn.Linear(config.n_embd, self.n_digits * 10)
        self.from_ohe = nn.Linear(10, config.n_embd)

    def forward(self, x):
        # x shape: (B, T, C)
        # 1) Perform the original Block forward pass
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # 2) Extract two tokens (for example, tokens 0 and 1) as n-digit numbers
        num1_embed = x[:, 0, :]  # shape (B, C)
        num2_embed = x[:, 1, :]  # shape (B, C)

        # 3) Convert each to OHE of shape (B, n_digits*10)
        num1_ohe = self.to_ohe(num1_embed)  # (B, n_digits*10)
        num2_ohe = self.to_ohe(num2_embed)  # (B, n_digits*10)

        # Reshape to (B, n_digits, 10), apply softmax digit-wise
        B = x.size(0)
        num1_ohe = num1_ohe.view(B, self.n_digits, 10)
        num1_ohe = Fnn.softmax(num1_ohe, dim=-1)  # encourage one-hot for each digit
        num1_ohe = num1_ohe.view(B, self.n_digits * 10)

        num2_ohe = num2_ohe.view(B, self.n_digits, 10)
        num2_ohe = Fnn.softmax(num2_ohe, dim=-1)
        num2_ohe = num2_ohe.view(B, self.n_digits * 10)

        # 4) Pass through F => (B, (n_digits + 1) * 10)
        sum_ohe = differentiable_addition(num1_ohe, num2_ohe, self.n_digits)

        # (Optional) again apply digit-wise softmax for the sum
        sum_ohe = sum_ohe.view(B, self.n_digits + 1, 10)
        sum_ohe = Fnn.softmax(sum_ohe, dim=-1)
        sum_ohe = sum_ohe.view(B, (self.n_digits + 1) * 10)

        # 5) Map each of the (n_digits + 1) digits from 10 -> embedding dimension
        sum_ohe_flat = sum_ohe.view(B * (self.n_digits + 1), 10)   # (B*(n_digits+1), 10)
        sum_embed_flat = self.from_ohe(sum_ohe_flat)               # (B*(n_digits+1), C)
        sum_embed = sum_embed_flat.view(B, self.n_digits + 1, -1)  # (B, n_digits+1, C)

        # 6) Concatenate to the end => (B, T + (n_digits+1), C)
        x = torch.cat([x, sum_embed], dim=1)
        return x


class FBlock_V2(Block):
    """
    A variant of the standard Transformer Block that:
      1) Performs the original block's attention + MLP,
      2) Learns to pick out two operand representations (via two trainable queries) from the entire sequence,
      3) Converts them to OHE, runs F, and appends the (n_digits+1) resulting tokens to the sequence.
    """
    def __init__(self, config):
        super().__init__(config)  # calls the original Block.__init__
        
        # We'll read n_digits from the config so we know how many digits per operand.
        self.n_digits = config.n_digits
        
        # Two queries that will attend over x to extract the two numbers
        # shape (config.n_embd,) each; we expand to (B, 1, C) at runtime
        self.query_1 = nn.Parameter(torch.randn(config.n_embd))
        self.query_2 = nn.Parameter(torch.randn(config.n_embd))
        
        # Projections for OHE:
        #   to_ohe:   from (C) -> (n_digits * 10)
        #   from_ohe: from (10) -> (C), reused digit-by-digit
        self.to_ohe   = nn.Linear(config.n_embd, self.n_digits * 10)
        self.from_ohe = nn.Linear(10, config.n_embd)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # 1) The normal Transformer block forward pass (from the original 'Block')
        #    - This lines up with the parent class structure: x + attn(...) + x + mlp(...)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # 2) Let each query vector attend over x to produce a single embedding for each operand
        B, T, C = x.shape
        
        # Expand queries to shape (B, 1, C) so we can do batch attention
        q_1 = self.query_1.view(1, 1, C).expand(B, 1, C)  # (B, 1, C)
        q_2 = self.query_2.view(1, 1, C).expand(B, 1, C)
        
        # We'll do a simple scaled dot-product between each query and every token
        # attn_wt_1 shape => (B, 1, T)
        attn_wt_1 = torch.einsum('bqc,btc->bqt', q_1, x) / (C ** 0.5)
        attn_wt_1 = Fnn.softmax(attn_wt_1, dim=-1)  # along T
        # Now multiply the attention weights by x to get a single vector
        z1 = torch.einsum('bqt,btc->bqc', attn_wt_1, x)  # (B, 1, C)
        z1 = z1.squeeze(1)  # (B, C)

        # Similarly for the second query
        attn_wt_2 = torch.einsum('bqc,btc->bqt', q_2, x) / (C ** 0.5)
        attn_wt_2 = Fnn.softmax(attn_wt_2, dim=-1)
        z2 = torch.einsum('bqt,btc->bqc', attn_wt_2, x)
        z2 = z2.squeeze(1)  # (B, C)

        # 3) Convert z1, z2 to one-hot-ish vectors => pass them to F => get sum
        num1_ohe = self.to_ohe(z1)  # (B, n_digits*10)
        num2_ohe = self.to_ohe(z2)  # (B, n_digits*10)

        # Reshape for digit-wise softmax => (B, n_digits, 10)
        num1_ohe = num1_ohe.view(B, self.n_digits, 10)
        num1_ohe = Fnn.softmax(num1_ohe, dim=-1)
        num1_ohe = num1_ohe.view(B, self.n_digits * 10)

        num2_ohe = num2_ohe.view(B, self.n_digits, 10)
        num2_ohe = Fnn.softmax(num2_ohe, dim=-1)
        num2_ohe = num2_ohe.view(B, self.n_digits * 10)

        # Pass through F => (B, (n_digits+1)*10) for the sum
        sum_ohe = differentiable_addition(num1_ohe, num2_ohe, self.n_digits)

        # Enforce digit-wise softmax again => (B, (n_digits+1), 10)
        sum_ohe = sum_ohe.view(B, self.n_digits + 1, 10)
        sum_ohe = Fnn.softmax(sum_ohe, dim=-1)
        sum_ohe = sum_ohe.view(B, (self.n_digits + 1) * 10)

        # 4) Map each digit from 10 -> embedding dimension
        sum_ohe_flat = sum_ohe.view(B * (self.n_digits + 1), 10)  # (B*(n_digits+1), 10)
        sum_embed_flat = self.from_ohe(sum_ohe_flat)              # (B*(n_digits+1), C)
        sum_embed = sum_embed_flat.view(B, self.n_digits + 1, C)  # (B, n_digits+1, C)

        # 5) Concatenate these new tokens onto the sequence => (B, T + (n_digits+1), C)
        x = torch.cat([x, sum_embed], dim=1)
        
        # Return x for the next layers
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
