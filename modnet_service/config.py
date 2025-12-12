"""
Configuration loader for the MODNet background-removal service.

Environment variables are centralized here to keep the rest of the code
focused on business logic and to make operational tuning clear.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model + preprocessing
    modnet_model_path: Path = Field(..., env="MODNET_MODEL_PATH")
    modnet_max_long_edge: int = Field(1024, env="MODNET_MAX_LONG_EDGE")
    modnet_max_long_edge_high_quality: int = Field(
        1536, env="MODNET_MAX_LONG_EDGE_HIGH_QUALITY"
    )
    default_quality_mode: str = Field("standard", env="DEFAULT_QUALITY_MODE")

    # Cloudflare R2 / S3-compatible storage
    r2_endpoint: Optional[str] = Field(None, env="R2_ENDPOINT")
    r2_access_key_id: Optional[str] = Field(None, env="R2_ACCESS_KEY_ID")
    r2_secret_access_key: Optional[str] = Field(None, env="R2_SECRET_ACCESS_KEY")
    r2_bucket_name: Optional[str] = Field(None, env="R2_BUCKET_NAME")
    r2_public_base_url: Optional[str] = Field(None, env="R2_PUBLIC_BASE_URL")

    # API
    request_timeout_seconds: int = Field(30, env="REQUEST_TIMEOUT_SECONDS")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Post-processing tunables
    edge_smooth_blend: float = Field(0.55, env="EDGE_SMOOTH_BLEND")
    edge_band_low: float = Field(0.08, env="EDGE_BAND_LOW")
    edge_band_high: float = Field(0.92, env="EDGE_BAND_HIGH")
    bilateral_sigma_color: float = Field(28.0, env="BILATERAL_SIGMA_COLOR")
    alpha_band_pull: float = Field(0.015, env="ALPHA_BAND_PULL")
    defringe_blend: float = Field(0.35, env="DEFRINGE_BLEND")
    postproc_mode: str = Field("portrait", env="POSTPROC_MODE")
    alpha_high_clip: float = Field(0.995, env="ALPHA_HIGH_CLIP")
    cc_keep_threshold: float = Field(0.05, env="CC_KEEP_THRESHOLD")

    # Debugging
    debug: bool = Field(False, env="DEBUG")
    debug_output_dir: Path = Field(Path("/tmp/bgremover_debug"), env="DEBUG_OUTPUT_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("default_quality_mode")
    def validate_quality_mode(cls, v: str) -> str:  # noqa: B902
        if v not in {"fast", "standard", "high"}:
            raise ValueError("DEFAULT_QUALITY_MODE must be one of fast|standard|high")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings to avoid reparsing env on every call."""
    settings = Settings()
    if not settings.modnet_model_path:
        raise ValueError("MODNET_MODEL_PATH is required")
    return settings


def quality_to_long_edge(quality_mode: str, settings: Optional[Settings] = None) -> int:
    """
    Translate a quality string into the resize target for the longest edge.

    Higher values give finer detail at the cost of speed/memory.
    """
    settings = settings or get_settings()
    if quality_mode == "high":
        return settings.modnet_max_long_edge_high_quality
    return settings.modnet_max_long_edge
