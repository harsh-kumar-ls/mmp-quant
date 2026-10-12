"""RapidOCR backend — PP-OCRv4 weights via ONNXRuntime."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from ...types import OcrLine
from .base import OCRBackend, OcrFrameResult, register

log = logging.getLogger(__name__)


@register
class RapidOCRBackend(OCRBackend):
    name = "rapidocr"

    @functools.cached_property
    def _engine(self):  # pragma: no cover - loads ~15 MB of ONNX weights
        from rapidocr_onnxruntime import RapidOCR

        log.info("initialising RapidOCR (PP-OCRv4 / ONNXRuntime)")
        return RapidOCR()

    def read(self, image_path: Path) -> OcrFrameResult:  # pragma: no cover
        result, _elapse = self._engine(str(image_path))
        if not result:
            return OcrFrameResult(lines=(), n_detected=0, n_rejected=0)

        lines: list[tuple[tuple[float, float, float, float], OcrLine]] = []
        rejected: list[float] = []
        for item in result:
            box, text, score = item[0], item[1], item[2]
            text = str(text).strip()
            confidence = float(score)
            if not text or confidence < self.min_confidence:
                # Recorded, not silently dropped: a region the detector found
                # but the recogniser could not read is evidence that text is
                # present and unreadable, which is not the same as absent.
                rejected.append(confidence)
                continue
            bbox = _axis_aligned(box)
            lines.append((bbox, OcrLine(text=text, confidence=confidence, bbox=bbox)))

        return OcrFrameResult(
            lines=tuple(line for _, line in sorted(lines, key=_reading_order)),
            n_detected=len(result),
            n_rejected=len(rejected),
            rejected_confidence_mean=(
                sum(rejected) / len(rejected) if rejected else None
            ),
        )


def _axis_aligned(box: Any) -> tuple[float, float, float, float]:
    """Collapse a 4-point quadrilateral to (x0, y0, x1, y1)."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return (min(xs), min(ys), max(xs), max(ys))


def _reading_order(entry: tuple[tuple[float, float, float, float], OcrLine]) -> tuple[float, float]:
    """Sort detected lines into human reading order: top-to-bottom, left-to-right.

    This is not cosmetic. Verbatim overlap is ROUGE-L, and LCS is *order
    sensitive* — if OCR emits a slide's lines in detection order rather than
    reading order, the token sequence no longer matches the order the narrator
    speaks them in, and a word-for-word slide reading scores well below 1.0.

    Known limitation: a strict y-then-x sort interleaves the columns of a
    two-column slide. Proper handling needs column detection (e.g. clustering
    x-centres, then sorting within columns). Deferred rather than guessed at —
    it should be driven by how common multi-column layouts turn out to be in the
    actual corpus, and the per-window OCR dumps will show that.
    """
    (x0, y0, _x1, _y1), _line = entry
    return (y0, x0)
