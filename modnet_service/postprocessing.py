"""Post-processing for MODNet alpha mattes with edge-aware cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger(__name__)


MODE_PRESETS = {
    "portrait": {
        "edge_band_low": 0.08,
        "edge_band_high": 0.92,
        "edge_smooth_blend": 0.55,
        "alpha_band_pull": 0.015,
        "defringe_blend": 0.35,
        "alpha_high_clip": 0.995,
        "cc_keep_threshold": 0.05,
        "bilateral_sigma_color": 28.0,
    },
    "dress": {
        "edge_band_low": 0.04,
        "edge_band_high": 0.97,
        "edge_smooth_blend": 0.45,
        "alpha_band_pull": 0.006,
        "defringe_blend": 0.20,
        "alpha_high_clip": 0.999,
        "cc_keep_threshold": 0.05,
        "bilateral_sigma_color": 28.0,
    },
    "product": {
        "edge_band_low": 0.12,
        "edge_band_high": 0.88,
        "edge_smooth_blend": 0.65,
        "alpha_band_pull": 0.03,
        "defringe_blend": 0.45,
        "alpha_high_clip": 0.995,
        "cc_keep_threshold": 0.08,
        "bilateral_sigma_color": 28.0,
    },
}


@dataclass
class PostprocessOptions:
    postproc_mode: Optional[str] = None
    match_background: bool = False
    match_strength: float = 0.0  # 0..1
    bg_color: Optional[Tuple[int, int, int]] = None  # RGB
    bg_image: Optional[Image.Image] = None  # PIL RGB


MATCH_STRENGTH_CAP = 0.65


def _band_mask(alpha: np.ndarray, low: float, high: float) -> np.ndarray:
    return (alpha > low) & (alpha < high)


def _keep_largest_component(alpha: np.ndarray, threshold: float = 0.05) -> tuple[np.ndarray, int]:
    """Zero out all but the largest connected component above threshold."""
    mask = (alpha > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = max(num_labels - 1, 0)
    if num_labels <= 1:
        return alpha, component_count

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep = labels == largest_label
    return np.where(keep, alpha, 0.0), component_count


def _smooth_edge_band(
    alpha: np.ndarray,
    band_low: float,
    band_high: float,
    blend: float,
    bilateral_sigma_color: float,
) -> np.ndarray:
    """Limit smoothing to uncertain edge band to avoid halos."""
    band = _band_mask(alpha, band_low, band_high)
    if not np.any(band):
        return alpha

    alpha_out = alpha.copy()
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    median = cv2.medianBlur(alpha_u8, 3)
    bilateral = cv2.bilateralFilter(median, d=3, sigmaColor=bilateral_sigma_color, sigmaSpace=2)

    smooth = np.clip(bilateral.astype(np.float32) / 255.0, 0.0, 1.0)
    orig_band = alpha_out[band]
    smooth_band = smooth[band]
    alpha_out[band] = orig_band * (1.0 - blend) + smooth_band * blend
    return alpha_out


def _pull_edge_haze(alpha: np.ndarray, band_low: float, pull: float) -> np.ndarray:
    """Slightly contract low-opacity edge haze without hardening hair."""
    band = (alpha > band_low) & (alpha < 0.5)
    if not np.any(band):
        return alpha
    alpha_out = alpha.copy()
    alpha_out[band] = np.clip(
        alpha_out[band] - pull * (1.0 - alpha_out[band]),
        0.0,
        1.0,
    )
    return alpha_out


def _compute_reference_field(rgb: np.ndarray, mask: np.ndarray, kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Compute a color reference by blurring within a mask and normalizing."""
    mask = mask.astype(np.float32)
    mask_smoothed = cv2.blur(mask, kernel)
    ref = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        channel = rgb[..., c].astype(np.float32)
        weighted = cv2.blur(channel * mask, kernel)
        ref[..., c] = np.where(
            mask_smoothed > 1e-3,
            weighted / np.clip(mask_smoothed, 1e-3, None),
            channel,
        )
    return ref


def _compute_references(rgb: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute solid foreground and source background reference fields."""
    fg_mask = alpha > 0.98
    fg_reference = _compute_reference_field(rgb, fg_mask.astype(np.float32))

    bg_mask = alpha < 0.02
    if np.any(bg_mask):
        # Dilate to sample a ring of true background, avoiding hair wisps.
        bg_mask = cv2.dilate(bg_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    bg_reference = _compute_reference_field(rgb, bg_mask.astype(np.float32))
    return fg_reference, bg_reference


def _apply_background_match(
    rgb: np.ndarray,
    alpha: np.ndarray,
    band_low: float,
    band_high: float,
    match_strength: float,
    bg_dst: Optional[np.ndarray],
    bg_src: np.ndarray,
) -> np.ndarray:
    """Nudge edge colors toward the target background to reduce paste-on look."""
    if bg_dst is None or match_strength <= 0:
        return rgb

    match_strength = float(np.clip(match_strength, 0.0, MATCH_STRENGTH_CAP))
    band = _band_mask(alpha, band_low, band_high)
    if not np.any(band):
        return rgb

    rgb_f = rgb.astype(np.float32)
    bg_src = bg_src.astype(np.float32)
    bg_dst = bg_dst.astype(np.float32)

    w = (1.0 - alpha) * match_strength
    w = np.where(band, w, 0.0)
    shift = w[..., None] * (bg_dst - bg_src)
    return np.clip(rgb_f + shift, 0.0, 255.0).astype(np.uint8)


def _defringe_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    band_low: float,
    band_high: float,
    blend_strength: float,
    fg_reference: np.ndarray,
) -> np.ndarray:
    """Blend edge pixels toward nearby solid foreground colors to reduce color bleed."""
    band = _band_mask(alpha, band_low, band_high)
    if not np.any(band):
        return rgb

    rgb_f = rgb.astype(np.float32)
    rgb_f = np.where(
        band[..., None],
        rgb_f * (1.0 - blend_strength) + fg_reference * blend_strength,
        rgb_f,
    )
    return np.clip(rgb_f, 0.0, 255.0).astype(np.uint8)


def _maybe_dump_debug(rgb: np.ndarray, alpha_u8: np.ndarray, band: np.ndarray, debug_dir: Path) -> None:
    """Optionally write debug visualizations when DEBUG is enabled."""
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        alpha_path = debug_dir / "alpha.png"
        overlay_path = debug_dir / "band_overlay.png"

        cv2.imwrite(str(alpha_path), alpha_u8)

        overlay = rgb.copy()
        overlay[band] = [255, 0, 0]  # highlight band pixels in red (RGB)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        logger.debug("postprocess: wrote debug outputs to %s", debug_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("postprocess: failed to write debug outputs: %s", exc)


def refine_alpha_and_compose_rgba(
    rgb_image: Image.Image,
    alpha_raw: np.ndarray,
    quality_mode: str,
    options: Optional[PostprocessOptions] = None,
) -> bytes:
    """Post-process alpha matte for portrait-friendly cutouts."""
    settings = config.get_settings()
    options = options or PostprocessOptions()
    preset_name = (options.postproc_mode or settings.postproc_mode).lower()
    preset = {} if preset_name == "custom" else MODE_PRESETS.get(preset_name, MODE_PRESETS["portrait"])
    if preset_name not in MODE_PRESETS and preset_name != "custom":
        logger.warning("postprocess: unknown mode '%s', falling back to portrait", preset_name)

    # Mode parameters
    if preset_name == "custom":
        band_low = float(np.clip(settings.edge_band_low, 0.0, 1.0))
        band_high = float(np.clip(settings.edge_band_high, 0.0, 1.0))
        blend = float(np.clip(settings.edge_smooth_blend, 0.0, 1.0))
        bilateral_sigma_color = float(max(settings.bilateral_sigma_color, 0.0))
        alpha_band_pull = float(max(settings.alpha_band_pull, 0.0))
        defringe_blend = float(np.clip(settings.defringe_blend, 0.0, 1.0))
        alpha_high_clip = float(np.clip(settings.alpha_high_clip, 0.0, 1.0))
        cc_keep_threshold = float(np.clip(settings.cc_keep_threshold, 0.0, 1.0))
    else:
        band_low = preset.get("edge_band_low", 0.08)
        band_high = preset.get("edge_band_high", 0.92)
        blend = preset.get("edge_smooth_blend", 0.55)
        alpha_band_pull = preset.get("alpha_band_pull", 0.015)
        defringe_blend = preset.get("defringe_blend", 0.35)
        alpha_high_clip = preset.get("alpha_high_clip", 0.995)
        cc_keep_threshold = preset.get("cc_keep_threshold", 0.05)
        bilateral_sigma_color = preset.get("bilateral_sigma_color", 28.0)

    match_background = bool(options.match_background)
    match_strength = float(np.clip(options.match_strength, 0.0, 1.0))
    if not match_background and match_strength > 0 and (options.bg_image is not None or options.bg_color is not None):
        match_background = True

    band_low = float(np.clip(band_low, 0.0, 1.0))
    band_high = float(np.clip(band_high, 0.0, 1.0))
    if band_high <= band_low:
        band_low, band_high = 0.05, 0.95

    rgb_np = np.array(rgb_image).astype(np.uint8)

    bg_dst = None
    if match_background:
        if options.bg_image is not None:
            bg_dst = np.array(options.bg_image.convert("RGB")).astype(np.uint8)
        elif options.bg_color is not None:
            bg_dst = np.zeros_like(rgb_np, dtype=np.uint8)
            bg_dst[..., 0] = options.bg_color[0]
            bg_dst[..., 1] = options.bg_color[1]
            bg_dst[..., 2] = options.bg_color[2]
        if bg_dst is not None and bg_dst.shape[:2] != rgb_np.shape[:2]:
            bg_dst = cv2.resize(bg_dst, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        if bg_dst is None:
            match_background = False

    alpha = np.clip(alpha_raw, 0.0, 1.0)
    alpha = np.power(alpha, 1.25)
    alpha = np.where(alpha < 0.03, 0.0, alpha)
    alpha = np.where(alpha > alpha_high_clip, 1.0, alpha)

    logger.debug(
        "postprocess mode=%s band_low=%.3f band_high=%.3f blend=%.3f bilat_sigma=%.1f pull=%.3f defringe=%.2f clamp_hi=%.3f cc_thresh=%.2f match_bg=%s match_strength=%.3f",
        preset_name,
        band_low,
        band_high,
        blend,
        bilateral_sigma_color,
        alpha_band_pull,
        defringe_blend,
        alpha_high_clip,
        cc_keep_threshold,
        match_background,
        match_strength,
    )

    cc_mask = (alpha > cc_keep_threshold).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(cc_mask, connectivity=8)
    component_count = max(num_labels - 1, 0)
    logger.debug(
        "postprocess: connected components above %.3f alpha=%d", cc_keep_threshold, component_count
    )

    alpha, _ = _keep_largest_component(alpha, threshold=cc_keep_threshold)

    band_before = _band_mask(alpha, band_low, band_high)
    band_fraction = float(np.mean(band_before)) if band_before.size else 0.0
    band_mean_before = float(alpha[band_before].mean()) if np.any(band_before) else 0.0
    logger.debug(
        "postprocess: band fraction=%.4f (%.2f%%), mean alpha before smooth=%.4f",
        band_fraction,
        band_fraction * 100.0,
        band_mean_before,
    )

    alpha = _smooth_edge_band(
        alpha,
        band_low=band_low,
        band_high=band_high,
        blend=blend,
        bilateral_sigma_color=bilateral_sigma_color,
    )

    band_after_smooth = _band_mask(alpha, band_low, band_high)
    band_mean_after_smooth = (
        float(alpha[band_after_smooth].mean()) if np.any(band_after_smooth) else 0.0
    )
    logger.debug(
        "postprocess: mean alpha after smooth=%.4f", band_mean_after_smooth
    )

    if alpha_band_pull > 0:
        alpha = _pull_edge_haze(alpha, band_low=band_low, pull=alpha_band_pull)

    band_after_pull = _band_mask(alpha, band_low, band_high)
    band_mean_after_pull = (
        float(alpha[band_after_pull].mean()) if np.any(band_after_pull) else 0.0
    )
    logger.debug(
        "postprocess: mean alpha after pull=%.4f", band_mean_after_pull
    )

    fg_reference, bg_reference = _compute_references(rgb_np, alpha)

    rgb_np = _defringe_rgb(
        rgb_np,
        alpha,
        band_low=band_low,
        band_high=band_high,
        blend_strength=defringe_blend,
        fg_reference=fg_reference,
    )

    if match_background and match_strength > 0 and bg_dst is not None:
        rgb_np = _apply_background_match(
            rgb_np,
            alpha,
            band_low=band_low,
            band_high=band_high,
            match_strength=match_strength,
            bg_dst=bg_dst,
            bg_src=bg_reference,
        )

    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    if settings.debug:
        band_final = _band_mask(alpha, band_low, band_high)
        _maybe_dump_debug(rgb_np, alpha_u8, band_final, Path(settings.debug_output_dir))

    rgba = np.dstack((rgb_np, alpha_u8))
    out = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
