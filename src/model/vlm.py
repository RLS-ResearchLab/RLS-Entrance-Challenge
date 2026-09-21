"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as model(images, input_ids), with
input_ids of shape (B, T), returning logits of shape (B, T_out, 27) where
T_out >= T and the last T logits line up with input_ids.
"""
