"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.
"""
