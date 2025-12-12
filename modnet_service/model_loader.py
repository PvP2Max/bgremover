"""
Model loading utilities for MODNet.

The loader:
 - builds the MODNet architecture,
 - loads the checkpoint from `MODNET_MODEL_PATH`,
 - keeps a single shared instance on GPU,
 - exposes `get_modnet_model()` for inference callers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Tuple

import torch

from . import config
from .modnet_model import MODNet

logger = logging.getLogger(__name__)

_MODEL = None
# Prefer CUDA -> Apple MPS -> CPU to support both GPU servers and local macOS dev.
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
    _DEVICE = torch.device("mps")
else:
    _DEVICE = torch.device("cpu")
_LOCK = Lock()


def get_device() -> torch.device:
    """Return the inference device (prefers CUDA when available)."""
    return _DEVICE


def _try_load_torchscript(model_path: Path) -> torch.nn.Module:
    """Load a TorchScript model if possible."""
    model = torch.jit.load(str(model_path), map_location=_DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    try:
        model.to(_DEVICE)
    except Exception:
        # TorchScript modules may not support .to; map_location already handled.
        pass
    return model


def _clean_state_dict(state_dict: dict) -> dict:
    """Remove common wrappers such as 'module.' prefixes."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def _load_modnet_from_state_dict(model_path: Path) -> torch.nn.Module:
    """Load a vanilla PyTorch checkpoint into the vendored MODNet architecture."""
    checkpoint = torch.load(model_path, map_location=_DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format for MODNet")

    checkpoint = _clean_state_dict(checkpoint)
    model = MODNet(backbone_pretrained=False)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        logger.warning("Missing keys when loading MODNet checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading MODNet checkpoint: %s", unexpected)

    model.to(_DEVICE)
    model.eval()
    return model


def _load_model(settings: config.Settings) -> torch.nn.Module:
    model_path = settings.modnet_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"MODNet checkpoint not found at {model_path}")

    try:
        logger.info("Attempting to load TorchScript model from %s", model_path)
        return _try_load_torchscript(model_path)
    except Exception as script_error:
        logger.info("TorchScript load failed, falling back to state_dict. Error: %s", script_error)
        return _load_modnet_from_state_dict(model_path)


def get_modnet_model() -> Tuple[torch.nn.Module, torch.device]:
    """
    Return a singleton MODNet model + device pair.

    The model is loaded once on first access and kept on GPU memory to
    avoid re-initialization costs across requests or batch jobs.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL, _DEVICE

    with _LOCK:
        if _MODEL is None:
            settings = config.get_settings()
            _MODEL = _load_model(settings)
            logger.info("MODNet loaded on device: %s", _DEVICE)
    return _MODEL, _DEVICE
