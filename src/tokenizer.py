"""Character-level tokenizer: 26 letters + <eos> (27 tokens).

Implements encode/decode between words and letter-index sequences, and
batches words of different lengths (padding + ignore_index for the loss).
"""
