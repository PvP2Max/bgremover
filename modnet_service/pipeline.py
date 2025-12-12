"""
High-level MODNet processing pipeline.

`process_image_bytes` is the main entry point used by both the HTTP API and
future batch/queue workers. It keeps orchestration simple:
bytes in -> preprocessing -> MODNet -> post-processing -> RGBA bytes out.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .model_loader import get_modnet_model
from .postprocessing import PostprocessOptions, refine_alpha_and_compose_rgba
from .preprocessing import PreprocessResult, load_and_preprocess_image_from_bytes

logger = logging.getLogger(__name__)


def _run_inference(preprocessed: PreprocessResult, model: torch.nn.Module) -> np.ndarray:
    """Run MODNet to produce an alpha matte in the original resolution."""
    with torch.no_grad():
        _, _, pred_matte = model(preprocessed.tensor, inference=True)  # (B,1,H,W)
    matte = F.interpolate(
        pred_matte,
        size=(preprocessed.orig_size[1], preprocessed.orig_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    alpha = matte[0, 0].detach().cpu().numpy()
    return np.clip(alpha, 0.0, 1.0)


def process_image_bytes(
    image_bytes: bytes,
    quality_mode: str = "standard",
    postprocess_options: Optional[PostprocessOptions] = None,
) -> bytes:
    """
    Full pipeline from raw bytes to RGBA PNG bytes.

    Raises:
        ValueError: when input is invalid or processing fails.
    """
    settings = config.get_settings()
    quality_mode = quality_mode or settings.default_quality_mode
    if quality_mode not in {"fast", "standard", "high"}:
        raise ValueError("qualityMode must be one of fast | standard | high")

    model, device = get_modnet_model()
    max_edge = config.quality_to_long_edge(quality_mode, settings=settings)

    preprocessed = load_and_preprocess_image_from_bytes(image_bytes, max_edge, device)
    alpha = _run_inference(preprocessed, model)

    png_bytes = refine_alpha_and_compose_rgba(
        rgb_image=preprocessed.original_image,
        alpha_raw=alpha,
        quality_mode=quality_mode,
        options=postprocess_options,
    )
    return png_bytes
