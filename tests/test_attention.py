"""Equivalence test for the handwritten attention implementation.

Compares src/model/attention.py against F.scaled_dot_product_attention
on random inputs with the same mask; outputs should match to about
1e-5 in float32.
"""
