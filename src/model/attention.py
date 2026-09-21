"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.

The provided tests expect the last nn.Module defined in this file to be
Cls(d_model, n_heads), called as attention(x, mask) with x of shape
(B, T, d_model). `mask` is optional; when given it is a boolean tensor where
True means "this query may attend to this key".
"""
