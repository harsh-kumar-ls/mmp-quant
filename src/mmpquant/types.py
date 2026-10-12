"""Data types passed between pipeline stages.

Everything here is frozen and JSON-serialisable via `to_dict()`. Stages
communicate only through these types, so any stage can be swapped or replayed
from a logged artifact without touching the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AsrWord:
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str
    words: tuple[AsrWord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass(frozen=True)
class Transcript:
    segments: tuple[AsrSegment, ...]
    language: str | None
    language_probability: float | None
    duration: float

    @property
    def words(self) -> tuple[AsrWord, ...]:
        return tuple(w for s in self.segments for w in s.words)

    @property
    def is_silent(self) -> bool:
        return len(self.segments) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "n_segments": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
        }


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    # (x0, y0, x1, y1) axis-aligned bounding box in pixels
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "bbox": self.bbox}


@dataclass(frozen=True)
class OcrBlock:
    """OCR result for one *visually distinct* stretch of video.

    Frame-diff deduplication collapses a run of near-identical sampled frames
    into a single block, so a static slide shown for 40 s produces one block
    covering [start, end) rather than 80 identical OCR results at 2 fps.
    """

    start: float
    end: float
    lines: tuple[OcrLine, ...]
    source_frame_time: float  # timestamp of the frame actually sent to OCR
    n_frames_collapsed: int  # how many sampled frames this block represents

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "source_frame_time": self.source_frame_time,
            "n_frames_collapsed": self.n_frames_collapsed,
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass(frozen=True)
class OcrTrack:
    blocks: tuple[OcrBlock, ...]
    n_frames_sampled: int

    @property
    def is_text_free(self) -> bool:
        return all(not b.lines for b in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frames_sampled": self.n_frames_sampled,
            "n_blocks": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
        }


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Window:
    """One time window with the narration and on-screen text aligned to it."""

    index: int
    start: float
    end: float

    asr_text: str
    ocr_text: str

    # provenance, for the audit trail
    asr_pool_start: float
    asr_pool_end: float
    ocr_block_indices: tuple[int, ...] = ()

    # On-screen text after layout reconstruction. `ocr_sentence_text` holds only
    # the elements eligible to constitute redundancy; keywords, titles and
    # labels are excluded from it. Scoring uses this, never `ocr_text`.
    # `ocr_text` is retained in full as the audit record of what was read.
    ocr_sentence_text: str = ""
    ocr_elements: tuple[dict[str, Any], ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "asr_text": self.asr_text,
            "ocr_text": self.ocr_text,
            "ocr_sentence_text": self.ocr_sentence_text,
            "ocr_elements": list(self.ocr_elements),
            "asr_pool_start": self.asr_pool_start,
            "asr_pool_end": self.asr_pool_end,
            "ocr_block_indices": list(self.ocr_block_indices),
        }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowScore:
    """Score for a single window, plus everything needed to justify it."""

    window: Window
    score: float | None  # 0-100, or None if the window does not qualify
    qualifies: bool
    components: dict[str, float] = field(default_factory=dict)
    exclusion_reason: str | None = None
    # Why the window scored what it did, in words a reader can check: e.g. the
    # content words the narration shared with the screen, and how many
    # narration elsewhere in the video shares by chance.
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.window.to_dict(),
            "score": self.score,
            "qualifies": self.qualifies,
            "components": self.components,
            "exclusion_reason": self.exclusion_reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class PrincipleResult:
    """Video-level result for one principle.

    `score is None` means N/A. It NEVER means zero, and must not be coerced to
    zero anywhere downstream — a video with no qualifying windows is unscoreable,
    not perfectly non-redundant.
    """

    principle: str
    score: float | None
    status: str  # "OK" | "N/A"
    na_reason: str | None
    window_scores: tuple[WindowScore, ...]
    n_windows_total: int
    n_windows_qualifying: int
    params: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle": self.principle,
            "score": self.score,
            "status": self.status,
            "na_reason": self.na_reason,
            "n_windows_total": self.n_windows_total,
            "n_windows_qualifying": self.n_windows_qualifying,
            "params": self.params,
            "warnings": list(self.warnings),
            "windows": [w.to_dict() for w in self.window_scores],
        }
