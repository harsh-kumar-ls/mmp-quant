"""OCR backends and frame deduplication."""

from . import apple, rapid  # noqa: F401  -- registers the backends
from .base import (  # noqa: F401
    RELIABILITY_NO_TEXT,
    RELIABILITY_OK,
    RELIABILITY_UNREADABLE,
    OCRBackend,
    OcrFrameResult,
    assess_reading_quality,
    assess_recognition_reliability,
    available,
    get_backend,
    register,
)
