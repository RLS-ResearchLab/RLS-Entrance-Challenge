"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.
"""
