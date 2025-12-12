"""
FastAPI layer exposing MODNet inference.

Endpoints:
 - GET /health
 - POST /remove-bg
"""

from __future__ import annotations

from io import BytesIO
import logging
import uuid
from typing import Optional
from urllib.parse import urljoin

import boto3
from botocore.client import Config as BotoConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image

from . import config
from .pipeline import process_image_bytes
from .postprocessing import PostprocessOptions

settings = config.get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="MODNet Background Removal Service", version="0.1.0")


class RemoveBgRequest(BaseModel):
    imageUrl: HttpUrl
    qualityMode: Optional[str] = None
    postprocessMode: Optional[str] = None
    matchBackground: Optional[bool] = None
    matchStrength: Optional[float] = None  # 0-1 or 0-100 slider
    backgroundColor: Optional[str] = None  # "#RRGGBB"
    backgroundImageUrl: Optional[HttpUrl] = None


class RemoveBgResponse(BaseModel):
    outputUrl: HttpUrl
    mode: str


def _get_s3_client():
    required = [
        settings.r2_endpoint,
        settings.r2_access_key_id,
        settings.r2_secret_access_key,
        settings.r2_bucket_name,
    ]
    if any(v is None for v in required):
        raise RuntimeError("R2 configuration is incomplete; check env vars.")
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        endpoint_url=settings.r2_endpoint,
        config=BotoConfig(signature_version="s3v4"),
    )


def _build_public_url(key: str) -> str:
    if settings.r2_public_base_url:
        return urljoin(settings.r2_public_base_url.rstrip("/") + "/", key)
    # Fallback: virtual-hosted-style may not be available; presigned URLs are safer
    client = _get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket_name, "Key": key},
        ExpiresIn=3600,
    )


def _parse_hex_color(value: Optional[str]) -> Optional[tuple[int, int, int]]:
    if not value:
        return None
    raw = value.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) != 6:
        return None
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
        return (r, g, b)
    except Exception:  # noqa: BLE001
        return None


def _download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=(5, settings.request_timeout_seconds))
    resp.raise_for_status()
    return resp.content


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-bg", response_model=RemoveBgResponse)
def remove_bg(body: RemoveBgRequest):
    try:
        image_bytes = _download_image(str(body.imageUrl))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to download image: %s", exc)
        raise HTTPException(status_code=400, detail="Could not download image") from exc

    bg_image_pil: Optional[Image.Image] = None
    if body.backgroundImageUrl:
        try:
            bg_bytes = _download_image(str(body.backgroundImageUrl))
            bg_image_pil = Image.open(BytesIO(bg_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to download background image: %s", exc)
            raise HTTPException(status_code=400, detail="Could not download background image") from exc

    match_strength = float(body.matchStrength or 0.0)
    if match_strength > 1.0:
        match_strength = match_strength / 100.0
    match_strength = min(max(match_strength, 0.0), 1.0)
    bg_color = _parse_hex_color(body.backgroundColor)
    match_background = bool(body.matchBackground) if body.matchBackground is not None else False
    if (bg_color or bg_image_pil) and match_strength > 0 and not match_background:
        match_background = True

    postprocess_options = PostprocessOptions(
        postproc_mode=body.postprocessMode,
        match_background=match_background,
        match_strength=match_strength,
        bg_color=bg_color,
        bg_image=bg_image_pil,
    )

    try:
        png_bytes = process_image_bytes(
            image_bytes,
            quality_mode=body.qualityMode or settings.default_quality_mode,
            postprocess_options=postprocess_options,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:  # noqa: BLE001
        logger.exception("MODNet processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Background removal failed") from exc

    key = f"modnet/{uuid.uuid4()}.png"
    try:
        client = _get_s3_client()
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=key,
            Body=png_bytes,
            ContentType="image/png",
        )
        output_url = _build_public_url(key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to upload cutout to R2: %s", exc)
        raise HTTPException(status_code=500, detail="Upload to storage failed") from exc

    return RemoveBgResponse(outputUrl=output_url, mode=body.qualityMode or settings.default_quality_mode)
