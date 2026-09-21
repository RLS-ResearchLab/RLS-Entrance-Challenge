"""CNN vision encoder.

Built from nn.Conv2d, activations, normalization and pooling (no
pretrained backbones). Turns a (B, 3, 64, 64) image into a feature map
(B, C, H', W').

The provided tests expect the last nn.Module defined in this file to be
constructible with no arguments and called as encoder(images).
"""
