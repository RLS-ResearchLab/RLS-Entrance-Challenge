"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index,
backward pass, optimizer step, periodic validation, and checkpointing.
"""
