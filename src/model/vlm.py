"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.
"""
