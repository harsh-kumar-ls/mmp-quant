"""Apple Vision backend — macOS's built-in text recogniser (VNRecognizeTextRequest).

Adopted 04 Oct 2026 for handwriting. The default PP-OCR models are trained on
printed text and read chalk confidently and wrongly ("electromagnetism" ->
"electmmoquehisin" on Test_22, "length time mass" -> "lengfh tim mn" on
Test_20). Apple's recogniser is trained on handwriting as well as print, runs
locally, and sends nothing off the machine.

The cost is reproducibility: it exists only on macOS, and its model is updated
with the operating system, so the same frame can read differently on another
macOS version. The OS version is therefore recorded in `describe()`, which
lands in every run's ocr.json. RapidOCR stays registered so the two can be
compared on the same frames, and so the pipeline still runs elsewhere.

Rejection works as for RapidOCR: an observation whose best reading falls below
`min_confidence` is counted as detected-but-rejected rather than dropped, so
the reliability check can still tell "unreadable" from "absent".
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

from ...types import OcrLine
from .base import OCRBackend, OcrFrameResult, register

log = logging.getLogger(__name__)


@register
class AppleVisionBackend(OCRBackend):
    name = "apple"

    def __init__(self, min_confidence: float = 0.5, language_correction: bool = True) -> None:
        super().__init__(min_confidence=min_confidence)
        self.language_correction = language_correction

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "language_correction": self.language_correction,
            "recognition_level": "accurate",
            "macos_version": platform.mac_ver()[0] or "not macOS",
        }

    def read(self, image_path: Path) -> OcrFrameResult:  # pragma: no cover - macOS only
        import Vision
        from Foundation import NSURL
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(self.language_correction)
        request.setRecognitionLanguages_(["en-US"])

        url = NSURL.fileURLWithPath_(str(image_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Apple Vision failed on {image_path}: {error}")

        observations = request.results() or []
        lines: list[OcrLine] = []
        rejected: list[float] = []
        for obs in observations:
            candidates = obs.topCandidates_(1)
            if not candidates:
                rejected.append(0.0)
                continue
            text = str(candidates[0].string()).strip()
            confidence = float(candidates[0].confidence())
            if not text or confidence < self.min_confidence:
                rejected.append(confidence)
                continue
            lines.append(
                OcrLine(text=text, confidence=confidence,
                        bbox=_to_pixels(obs.boundingBox(), width, height))
            )

        lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]))  # reading order
        return OcrFrameResult(
            lines=tuple(lines),
            n_detected=len(observations),
            n_rejected=len(rejected),
            rejected_confidence_mean=(sum(rejected) / len(rejected) if rejected else None),
        )


def _to_pixels(rect: Any, width: int, height: int) -> tuple[float, float, float, float]:
    """Vision's normalised, bottom-left-origin box -> pixel (x0, y0, x1, y1), top-left origin.

    The layout grouping in onscreen.py measures gaps and alignment in pixels
    with y growing downward, as RapidOCR reports them; Vision reports fractions
    of the image with y growing upward. Converted here so the two backends are
    interchangeable downstream.
    """
    x, y = rect.origin.x, rect.origin.y
    w, h = rect.size.width, rect.size.height
    return (x * width, (1.0 - y - h) * height, (x + w) * width, (1.0 - y) * height)
