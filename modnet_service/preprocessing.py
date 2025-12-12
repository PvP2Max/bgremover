"""
Image loading and preprocessing for MODNet.

The preprocessing normalizes images into MODNet's expected input space and
resizes by the longest edge for a speed/quality balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import math


@dataclass
class PreprocessResult:
    tensor: torch.Tensor
    original_image: Image.Image
    orig_size: Tuple[int, int]  # (width, height)
    resized_size: Tuple[int, int]


def _compute_resize_dims(width: int, height: int, max_long_edge: int) -> Tuple[int, int]:
    """Preserve aspect ratio while constraining the longest edge."""
    if max_long_edge <= 0:
        return width, height
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return width, height
    scale = max_long_edge / long_edge
    new_w = int(width * scale)
    new_h = int(height * scale)
    # MODNet down/up sampling chains work best when dimensions are divisible by 32.
    new_w = max(32, math.ceil(new_w / 32) * 32)
    new_h = max(32, math.ceil(new_h / 32) * 32)
    return new_w, new_h


def load_and_preprocess_image_from_bytes(
    image_bytes: bytes, max_long_edge: int, device: torch.device
) -> PreprocessResult:
    """
    Decode an image, resize longest edge to `max_long_edge`, and normalize.

    MODNet expects RGB inputs normalized to [-1, 1]. Resizing on the long edge
    keeps people large enough for hair detail while keeping inference fast.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid image data") from exc

    orig_w, orig_h = image.size
    new_w, new_h = _compute_resize_dims(orig_w, orig_h, max_long_edge)

    if (new_w, new_h) != (orig_w, orig_h):
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)
    else:
        image_resized = image

    im_np = np.asarray(image_resized).astype("float32") / 255.0
    # MODNet normalization: center to [-1, 1]
    im_np = (im_np - 0.5) / 0.5
    im_np = np.transpose(im_np, (2, 0, 1))  # HWC -> CHW

    tensor = torch.from_numpy(im_np).unsqueeze(0).to(device)

    return PreprocessResult(
        tensor=tensor,
        original_image=image,
        orig_size=(orig_w, orig_h),
        resized_size=(new_w, new_h),
    )
