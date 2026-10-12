"""OCR backend interface.

CLAUDE.md specifies PaddleOCR. This project runs RapidOCR by default, which
serves the *same* PP-OCRv4 detection and recognition weights through
ONNXRuntime instead of the PaddlePaddle framework. The models, and therefore the
accuracy characteristics and the citation, are unchanged; only the inference
runtime differs. The reason is platform reliability: PaddlePaddle's macOS/arm64
wheels are unreliable and bring a large framework dependency for no benefit
here, since we do no training.

The interface exists so that claim stays testable — a `PaddleOCRBackend` can be
dropped in and the two run side by side on the same frames to confirm the
substitution does not shift scores. Until that comparison is run, treat the
equivalence as an argued assumption, not a verified fact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ...types import OcrLine


@dataclass(frozen=True)
class OcrFrameResult:
    """What text recognition found in one frame, including what it rejected.

    Detection and recognition are separate stages: the model first locates text
    regions, then reads each one. Those fail independently, and the difference
    matters enormously.

      * nothing detected            -> the frame genuinely carries no text
      * detected but not readable   -> text IS present and we cannot read it

    Discarding rejected regions without recording them collapses those two into
    one, and they must never be collapsed. Handwriting is the case in point:
    PP-OCR models are trained on printed text, so a blackboard frame typically
    yields plenty of detections whose recognition is rejected. Treated as "no
    text on screen", a lecturer reading his own writing aloud -- maximum
    redundancy -- scores 0, which this project defines as ideal design.
    """

    lines: tuple[OcrLine, ...]
    n_detected: int
    n_rejected: int
    rejected_confidence_mean: float | None = None

    @property
    def detection_recovery_rate(self) -> float:
        """Fraction of detected regions that were successfully read."""
        if self.n_detected == 0:
            return 1.0  # nothing to recover; not a failure
        return len(self.lines) / self.n_detected

_REGISTRY: dict[str, type[OCRBackend]] = {}


def register(cls: type[OCRBackend]) -> type[OCRBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str, **kwargs: Any) -> OCRBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown OCR backend '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


class OCRBackend(ABC):
    name: ClassVar[str]

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    @abstractmethod
    def read(self, image_path: Path) -> OcrFrameResult:
        """Read one frame, reporting both what was recognised and what was not."""

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "min_confidence": self.min_confidence}


# --------------------------------------------------------------------------- #
# Reliability of text recognition across a whole video
# --------------------------------------------------------------------------- #

RELIABILITY_OK = "ok"
RELIABILITY_UNREADABLE = "text_present_but_unreadable"
RELIABILITY_NO_TEXT = "no_text_detected"


def assess_recognition_reliability(
    frame_results: list[OcrFrameResult],
    *,
    weights: Sequence[float] | None = None,
    min_recovery_rate: float = 0.5,
    min_frames_with_detections: float = 0.2,
) -> dict[str, Any]:
    """Decide whether the on-screen text was read well enough to score.

    Two independent quantities:

      * how often the detector found text at all;
      * of what it found, how much the recogniser could actually read.

    A video is flagged unreadable when text was detected across a meaningful
    share of the video but recovered from fewer than `min_recovery_rate` of those
    detections. That is the handwriting signature, and it is the one case where
    a score of 0 is not merely wrong but wrong in the flattering direction: a
    lecturer reading his own blackboard aloud is maximally redundant, and
    unreadable handwriting reports it as ideal design.

    Confidence alone cannot substitute for this. Recognition models return
    overconfident wrong readings, so a high mean confidence on the lines that
    did survive says nothing about the ones that did not.

    **"Share of the video" must be measured in time, not in results.** Each
    element of `frame_results` is one *deduplicated group*, and groups do not
    represent equal durations: a static board held for a minute is one group,
    while a minute of camera movement over the lecturer is a hundred. Counting
    groups therefore undercounts exactly the static, text-bearing content this
    check exists to catch. Pass `weights` -- the seconds each group covers -- and
    the share is computed over duration instead.

    Without weights the group count is used and the summary says so, because a
    caller that cannot supply durations should still get an answer; but the
    unweighted figure is not a statement about how much of the video showed text
    and must not be read as one.
    """
    if not frame_results:
        return {
            "reliability": RELIABILITY_NO_TEXT,
            "n_frames": 0,
            "frames_with_detections": 0,
            "detection_recovery_rate": None,
            "note": "no frames were read",
        }

    if weights is not None and len(weights) != len(frame_results):
        raise ValueError(
            f"weights has {len(weights)} entries for {len(frame_results)} results; "
            "each weight is the duration in seconds covered by the group at the "
            "same index"
        )

    n_frames = len(frame_results)
    with_detections = [f for f in frame_results if f.n_detected > 0]
    detected = sum(f.n_detected for f in frame_results)
    recovered = sum(len(f.lines) for f in frame_results)
    recovery = (recovered / detected) if detected else None

    share_by_group = len(with_detections) / n_frames
    share_by_duration: float | None = None
    if weights is not None:
        total = float(sum(weights))
        if total > 0:
            covered = sum(w for w, f in zip(weights, frame_results) if f.n_detected > 0)
            share_by_duration = covered / total

    # Duration is the honest basis; fall back only when the caller had none.
    basis = "duration" if share_by_duration is not None else "group_count"
    detection_share = (
        share_by_duration if share_by_duration is not None else share_by_group
    )

    summary: dict[str, Any] = {
        "n_frames": n_frames,
        "frames_with_detections": len(with_detections),
        "detection_share": round(detection_share, 4),
        "detection_share_basis": basis,
        "detection_share_by_group": round(share_by_group, 4),
        "detection_share_by_duration": (
            None if share_by_duration is None else round(share_by_duration, 4)
        ),
        "n_regions_detected": detected,
        "n_regions_recovered": recovered,
        "detection_recovery_rate": None if recovery is None else round(recovery, 4),
    }

    if detected == 0:
        summary["reliability"] = RELIABILITY_NO_TEXT
        summary["note"] = (
            "no text regions were detected in any frame; the video appears to "
            "carry no on-screen text at all"
        )
        return summary

    if detection_share >= min_frames_with_detections and recovery < min_recovery_rate:
        summary["reliability"] = RELIABILITY_UNREADABLE
        summary["note"] = (
            f"text was detected across {detection_share:.0%} of the video "
            f"(by {basis}) but only {recovery:.0%} of detected regions could be "
            "read. On-screen text is present and largely unreadable, which is the "
            "signature of handwriting or a low-contrast board. A low score here "
            "means the text could not be read, NOT that the video is free of "
            "redundancy. Do not report this score without inspecting frames."
        )
        return summary

    summary["reliability"] = RELIABILITY_OK
    summary["note"] = (
        f"{recovery:.0%} of detected text regions were read successfully"
    )
    return summary


# --------------------------------------------------------------------------- #
# Recognition accuracy: measured and recorded, deliberately NOT a verdict
# --------------------------------------------------------------------------- #


def assess_reading_quality(
    reliability: dict[str, Any],
    blocks: Sequence[Any],
) -> dict[str, Any]:
    """Record what share of the recognised words are real English words.

    Defect 10.3 (04 Oct 2026). `assess_recognition_reliability` catches OCR
    *refusing* to read. It cannot catch OCR reading confidently and wrongly --
    what chalk handwriting produces ("lengfh tim mn" for "length time mass" on
    Test_20). The obvious measure is the share of recognised words found in an
    English word list, weighted by time on screen (by duration, not block
    count -- the defect 9.1 lesson).

    It was built as a reliability verdict ("text present but misread", below a
    simple-majority 0.5) and measured against all 24 videos before being
    switched on. It does not separate. It caught two chalkboard videos (Test_22
    0.49, Test_23 0.41) but missed the clearest one (Test_20 0.55, where short
    misreads land on real words: "time" -> "tim", "speed" -> "shed"), and
    falsely flagged two cleanly-read printed videos -- test_1 (0.39), where OCR
    read correctly but dropped the spaces ("Heatcannotspontaneouslyflowfrom
    acolderbody"), and 14_JzbhQX00 (0.41), dominated by a video-editor
    watermark ("clideo") on screen for 186 s. Moving the threshold until those
    six land correctly would be fitting a rule to 24 videos.

    So the figure is recorded here for audit and future analysis, and changes
    no verdict. The word-level use of the same check -- a finite verb must be a
    real word, see onscreen._is_plausible_verb -- does not share these
    confounds and is in force.
    """
    from ...onscreen import real_word_share

    real_w = total_w = 0.0
    for b in blocks:
        duration = max(b.end - b.start, 0.0)
        text = "\n".join(line.text for line in b.lines)
        real, total = real_word_share(text)
        real_w += real * duration
        total_w += total * duration

    out = dict(reliability)
    out["real_word_share"] = None if total_w <= 0 else round(real_w / total_w, 4)
    return out
