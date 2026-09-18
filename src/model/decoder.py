"""Transformer decoder block and stack.

Masked self-attention + MLP + residual connections + LayerNorm, stacked
N times, with positional embeddings over the combined [visual tokens] +
[letter tokens] sequence.
"""
