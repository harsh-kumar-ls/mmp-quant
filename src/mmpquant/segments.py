"""Segment-level aggregation and blind annotation sheets.

Why segments exist as a separate layer
--------------------------------------
The supervisor will annotate at *segment* level, not video level (decision of
07 Sep 2026). The unit the system reports must therefore be the same unit a
human is asked to judge, or the two cannot be correlated at all.

10-second windows are the right granularity for *measurement* -- short enough
that narration and slide stay paired -- but the wrong granularity for *human
judgement*: asking a rater to score eighteen 10-second clips per video invites
fatigue and inconsistency. Segments sit above windows: a segment's score is the
duration-weighted mean of *every* window inside it, a window with nothing to
assess counting as 0 (see "One averaging rule" below).

One averaging rule, at every level (08 Oct 2026)
------------------------------------------------
Segment and video scores are both the share of viewing time spent on redundant
text: each window contributes `score × duration`, divided by the duration of
*all* windows in the span. A window with no on-screen sentence, or no
narration, contributes 0 -- there was nothing on screen for the narration to
duplicate, so for that stretch the viewer met no redundancy.

The rule before this one mixed two: inside a segment, empty windows were dropped
from the average; across segments, an empty segment counted as 0. The video
score then depended on where the 30 s grid happened to fall. Regrouping the same
window scores with the grid shifted by 10 s or 20 s moved Test_27 between 6.80
and 12.43, and Test_13 between 1.58 and 4.73 -- an artefact of segmentation,
not a property of the video. With one rule the video score is exactly the
duration-weighted mean of the segment scores *and* independent of the grid.

The average over assessed windows only ("when there is text on screen, how
redundant is it?") is kept beside it as `assessed_score`, together with
`coverage` (the share of time that could be assessed): score = assessed_score ×
coverage. Which of the two tracks the supervisor's judgement better is decided
on the tune split once ground truth exists -- these two, fixed in advance, and
no others.

Fixing the boundaries in advance
--------------------------------
The annotation sheet is generated from the same `length_s` the scorer uses, so
the rater's segments and the system's segments are identical by construction.
The alternative -- letting each rater mark their own spans while watching -- puts
the burden of reconciling mismatched boundaries onto the analysis, and there is
no honest way to do that after the fact.

The sheet deliberately contains no system output. A rater shown the machine's
score anchors to it, and the correlation that follows measures agreement with an
anchor rather than independent judgement.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import WindowScore


@dataclass(frozen=True)
class SegmentScore:
    index: int
    start: float
    end: float
    score: float | None
    status: str  # "OK" | "N/A"
    na_reason: str | None
    n_windows: int
    n_windows_qualifying: int
    window_indices: tuple[int, ...]
    # The two overlap measures, averaged over the same qualifying windows.
    # Reported alongside the blend so that the 0.6/0.4 weighting is not
    # load-bearing before it has been validated against ground truth: if the
    # combined score correlates worse than either component alone, that is
    # visible in the data rather than hidden inside the weights.
    verbatim: float | None = None
    semantic: float | None = None
    # Mean over assessed windows only (None if none was assessed), and the
    # share of the segment's measured time those windows cover.
    # score == assessed_score × coverage.
    assessed_score: float | None = None
    coverage: float = 0.0
    # Total duration of the windows inside: the weight this segment carries
    # in the video score (differs from `duration` only by a dropped tail).
    measured_s: float = 0.0
    # Every per-window component named by the caller, averaged with the same
    # denominator as the score: {"verbatim", "semantic"} for redundancy,
    # {"background_music", "off_topic"} for coherence. None if none assessed.
    parts: dict[str, float | None] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def timecode(self) -> str:
        def mmss(t: float) -> str:
            return f"{int(t) // 60:d}:{int(t) % 60:02d}"

        return f"{mmss(self.start)}-{mmss(self.end)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "timecode": self.timecode,
            "score": self.score,
            "status": self.status,
            "na_reason": self.na_reason,
            "n_windows": self.n_windows,
            "n_windows_qualifying": self.n_windows_qualifying,
            "window_indices": list(self.window_indices),
            "verbatim": self.verbatim,
            "semantic": self.semantic,
            "assessed_score": self.assessed_score,
            "coverage": self.coverage,
            "measured_s": self.measured_s,
            "parts": self.parts,
        }


# Matches extract.align.window_bounds' MIN_WINDOW_S: a tail this short holds
# nothing to measure or to judge.
MIN_SEGMENT_S = 0.1


def segment_bounds(duration: float, length_s: float) -> list[tuple[float, float]]:
    """Contiguous [start, end) segments covering [0, duration).

    A final segment shorter than `MIN_SEGMENT_S` is folded into the one before
    it. test9 is 180.098 s long: without this it got a seventh segment lasting
    0.098 s, which held no measurement window (windows drop tails that short),
    showed as N/A in every report, and put a "3:00-3:00" row on its annotation
    sheet for a rater to score.
    """
    if duration <= 0 or length_s <= 0:
        return []
    bounds: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        bounds.append((start, min(start + length_s, duration)))
        start += length_s
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_SEGMENT_S:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def aggregate_to_segments(
    window_scores: Sequence[WindowScore],
    duration: float,
    length_s: float,
    part_keys: Sequence[str] = ("verbatim", "semantic"),
) -> list[SegmentScore]:
    """Roll window scores up into segments.

    A window belongs to the segment containing its midpoint, so each window
    counts towards exactly one segment even when the segment length is not an
    exact multiple of the window length.

    Segment score = sum(score × duration over assessed windows) ÷ duration of
    all windows inside -- an unassessed window counts as 0 (module docstring,
    "One averaging rule"). A segment with no assessed window therefore scores
    0, not N/A: it was assessed and holds no redundancy. N/A is reserved for a
    segment holding no windows at all, which can only happen if windowing and
    segmentation disagree about the video's duration.
    """
    segments: list[SegmentScore] = []

    for index, (start, end) in enumerate(segment_bounds(duration, length_s)):
        inside = [
            ws
            for ws in window_scores
            if start <= (ws.window.start + ws.window.end) / 2.0 < end
        ]
        qualifying = [ws for ws in inside if ws.qualifies and ws.score is not None]

        if not inside:
            segments.append(
                SegmentScore(
                    index=index, start=start, end=end, score=None, status="N/A",
                    na_reason="no measurement windows fall inside this segment",
                    n_windows=0, n_windows_qualifying=0, window_indices=(),
                )
            )
            continue

        measured = sum(ws.window.duration for ws in inside)
        assessed = sum(ws.window.duration for ws in qualifying)
        score = sum(ws.score * ws.window.duration for ws in qualifying) / measured
        assessed_score = None
        parts: dict[str, float | None] = {k: None for k in part_keys}
        if qualifying:
            # Same denominator as the score, so e.g. score = 100 × (w_v·verbatim
            # + w_s·semantic) holds at segment level exactly as per window.
            parts = {
                k: round(sum(ws.components.get(k, 0.0) * ws.window.duration
                             for ws in qualifying) / measured, 4)
                for k in part_keys
            }
            assessed_score = round(
                sum(ws.score * ws.window.duration for ws in qualifying) / assessed, 2)

        segments.append(
            SegmentScore(
                index=index,
                start=start,
                end=end,
                score=round(score, 2),
                status="OK",
                na_reason=None,
                n_windows=len(inside),
                n_windows_qualifying=len(qualifying),
                window_indices=tuple(ws.window.index for ws in inside),
                verbatim=parts.get("verbatim"),
                semantic=parts.get("semantic"),
                assessed_score=assessed_score,
                coverage=round(assessed / measured, 4),
                measured_s=round(measured, 3),
                parts=parts,
            )
        )

    return segments


@dataclass(frozen=True)
class VideoAggregate:
    """The video-level numbers. `score` is the reported one."""

    score: float | None  # whole-video mean; unassessed windows count 0
    assessed_score: float | None  # mean over assessed windows only
    coverage: float  # share of measured time that was assessed


def aggregate_video(window_scores: Sequence[WindowScore]) -> VideoAggregate:
    """Whole-video score, computed from the windows directly.

    Identical to the duration-weighted mean of `aggregate_to_segments`' scores
    (weighted by `measured_s`) up to their 2-decimal rounding, because both use
    the same rule; computed here from the unrounded window scores so rounding
    is applied once, at the end.

    History: until 30 Sep 2026 the video score was the mean over assessed
    windows only (now `assessed_score`); `Test_13` got 21.48 from twenty
    seconds of a three-minute video. 30 Sep - 08 Oct it was the mean of
    segment scores under a mixed rule that made it depend on the segment grid.
    """
    measured = sum(ws.window.duration for ws in window_scores)
    if measured <= 0:
        return VideoAggregate(score=None, assessed_score=None, coverage=0.0)
    qualifying = [ws for ws in window_scores if ws.qualifies and ws.score is not None]
    assessed = sum(ws.window.duration for ws in qualifying)
    total = sum(ws.score * ws.window.duration for ws in qualifying)
    return VideoAggregate(
        score=round(total / measured, 2),
        assessed_score=round(total / assessed, 2) if assessed > 0 else None,
        coverage=round(assessed / measured, 4),
    )


def write_segment_scores(
    segments: Sequence[SegmentScore],
    video_id: str,
    path: Path,
    part_keys: Sequence[str] = ("verbatim", "semantic"),
) -> Path:
    """System scores per segment, in the shape the validation join expects.

    One file per principle (segment_scores.csv for redundancy,
    coherence_segment_scores.csv for coherence); the part columns differ.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["video_id", "segment_index", "start_s", "end_s", "timecode",
             "system_score", *part_keys, "status",
             "n_windows_qualifying", "assessed_score", "coverage"]
        )
        for s in segments:
            writer.writerow(
                [video_id, s.index, f"{s.start:.1f}", f"{s.end:.1f}", s.timecode,
                 "" if s.score is None else f"{s.score:.2f}",
                 *("" if s.parts.get(k) is None else f"{s.parts[k]:.4f}" for k in part_keys),
                 s.status, s.n_windows_qualifying,
                 "" if s.assessed_score is None else f"{s.assessed_score:.2f}",
                 f"{s.coverage:.4f}"]
            )
    return path


def write_annotation_sheet(
    video_id: str,
    duration: float,
    length_s: float,
    path: Path,
    raters: Sequence[str] = ("rater_1", "rater_2"),
) -> Path:
    """A blank sheet for human annotation, carrying no system output.

    One row per segment, one empty score column per rater. Boundaries come from
    the same `length_s` the scorer uses, so the returned annotations join to the
    system's segment scores on (video_id, segment_index) with no reconciliation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds = segment_bounds(duration, length_s)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["video_id", "segment_index", "start_s", "end_s", "timecode"]
        header += [f"{r}_score" for r in raters]
        header += [f"{r}_notes" for r in raters]
        writer.writerow(header)
        for index, (start, end) in enumerate(bounds):
            def mmss(t: float) -> str:
                return f"{int(t) // 60:d}:{int(t) % 60:02d}"

            writer.writerow(
                [video_id, index, f"{start:.1f}", f"{end:.1f}",
                 f"{mmss(start)}-{mmss(end)}"] + [""] * (2 * len(raters))
            )
    return path
