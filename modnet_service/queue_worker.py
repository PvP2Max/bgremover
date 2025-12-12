"""
Batch/queue worker stub.

Future BoothOS integration can pull jobs from Redis/Kafka/DB and reuse the
shared pipeline. The worker is intentionally minimal to keep it framework
agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

from .pipeline import process_image_bytes

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    input_url: str
    quality_mode: str = "standard"


def process_batch(items: Iterable[BatchItem]) -> List[bytes]:
    """
    Process a batch of images synchronously.

    Returns a list of PNG byte buffers matching the input order. Storage /
    queuing concerns are intentionally left to the caller so this can be
    embedded into any worker framework.
    """
    outputs: List[bytes] = []
    for item in items:
        logger.info("Processing batch item url=%s mode=%s", item.input_url, item.quality_mode)
        # Download logic intentionally omitted; callers can fetch bytes in
        # the format they prefer (local FS, signed URL, etc.).
        raise NotImplementedError(
            "Download/fetch layer is queue-specific. Provide image bytes and reuse `process_image_bytes`."
        )
    return outputs

