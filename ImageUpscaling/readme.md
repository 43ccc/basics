# Upscaler
Basic upscaler to increase images sizes. Images can be extracted from videos and pytorch resizing is used to downscale them to easily obtain a training set.

Expects 720p images, downscales them to 360p and learns to upscale them. Works, but performance is not that great, however this is mostly due to VRAM limiting the model.

Mixed precision is used, only a small batchsize and even with that it takes almost 8GB VRAM during training.

Adapted structure of https://arxiv.org/abs/1511.04587:

