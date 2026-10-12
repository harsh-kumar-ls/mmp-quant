"""Plugin interface for principle scorers.

The pipeline's extraction and alignment stages know nothing about Mayer's
principles — they produce `Window` objects. A scorer turns windows into a
`PrincipleResult`. Adding `coherence.py` later means writing one subclass and
registering it; no changes to extraction, alignment, reporting or the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..types import PrincipleResult, Transcript, Window, WindowScore


@dataclass(frozen=True)
class VideoContext:
    """Per-video inputs a scorer may need beyond the windows themselves.

    Redundancy needs only the windows. Coherence also needs the word-level
    transcript (which seconds are spoken) and the sound-event timeline (which
    seconds carry music). Passing them here keeps `Window` -- the alignment
    layer's output -- free of principle-specific fields.
    """

    transcript: Transcript | None = None
    audio_events: Any | None = None  # extract.audio_events.AudioEvents

_REGISTRY: dict[str, type[PrincipleScorer]] = {}


def register(cls: type[PrincipleScorer]) -> type[PrincipleScorer]:
    _REGISTRY[cls.name] = cls
    return cls


def get_scorer(name: str) -> type[PrincipleScorer]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


class PrincipleScorer(ABC):
    """Base class for a single-principle scorer.

    Contract:
      * `score_window` is pure and side-effect free — same window in, same
        score out. This is what makes the scorer unit-testable with synthetic
        text and no video.
      * A window that does not qualify returns `WindowScore(score=None,
        qualifies=False, exclusion_reason=...)`. It is never scored 0 and never
        silently dropped: the excluded windows and their reasons are part of the
        audit trail.
      * A video with no qualifying windows yields `score=None, status="N/A"`.
        Coercing that to 0 anywhere downstream is a bug.
    """

    name: ClassVar[str]

    @abstractmethod
    def score_window(self, window: Window) -> WindowScore:
        ...

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The exact parameters this run used, for the methodology appendix."""

    def score(
        self,
        windows: Sequence[Window],
        extra_warnings: Iterable[str] = (),
        context: VideoContext | None = None,
    ) -> PrincipleResult:
        window_scores = tuple(self.score_window(w) for w in windows)
        return self.aggregate(window_scores, extra_warnings=extra_warnings)

    def aggregate(
        self,
        window_scores: Sequence[WindowScore],
        extra_warnings: Iterable[str] = (),
    ) -> PrincipleResult:
        qualifying = [ws for ws in window_scores if ws.qualifies and ws.score is not None]
        warnings = list(extra_warnings)

        if not qualifying:
            # Two different situations look identical from the score alone, and
            # must not be collapsed together:
            #
            #   "we looked, and there is no redundancy here"  -> a real finding, 0
            #   "we could not look at all"                    -> a measurement
            #                                                    failure, N/A
            #
            # A video whose slides carry no text, or only short signaling labels,
            # genuinely cannot violate the redundancy principle: under Mayer that
            # is the recommended design, and 0 is the correct reading. Reporting
            # N/A there would also drop the video out of the validation
            # correlation entirely, discarding a real data point.
            #
            # A video we could not assess -- no speech anywhere, or no frames --
            # is different. Scoring it 0 would present a broken extraction as a
            # perfectly designed video, which is the failure mode the N/A rule
            # exists to prevent.
            unmeasurable = self._unmeasurable_reason(window_scores)
            if unmeasurable is not None:
                return PrincipleResult(
                    principle=self.name,
                    score=None,
                    status="N/A",
                    na_reason=unmeasurable,
                    window_scores=tuple(window_scores),
                    n_windows_total=len(window_scores),
                    n_windows_qualifying=0,
                    params=self.params(),
                    warnings=tuple(warnings),
                )

            warnings.append(
                "scored 0: the video was assessed successfully and contains no "
                f"qualifying on-screen text ({self._exclusion_breakdown(window_scores)}). "
                "This is the floor of the scale, not a failed measurement — verify "
                "against the OCR dump that the absence of text is real."
            )
            return PrincipleResult(
                principle=self.name,
                score=0.0,
                status="OK",
                na_reason=None,
                window_scores=tuple(window_scores),
                n_windows_total=len(window_scores),
                n_windows_qualifying=0,
                params=self.params(),
                warnings=tuple(warnings),
            )

        # Duration-weighted mean.
        #
        # NOTE: with fixed-size windows this is *almost* an unweighted mean --
        # only the partial final window carries a different weight. It is
        # implemented as a weighted mean anyway because (a) it is what the spec
        # defines, and (b) it stays correct if variable-length windowing (e.g.
        # slide-boundary or scene-cut segmentation) is adopted later.
        total_weight = sum(ws.window.duration for ws in qualifying)
        if total_weight <= 0:
            raise ValueError("qualifying windows have zero total duration")
        score = sum(ws.score * ws.window.duration for ws in qualifying) / total_weight

        coverage = total_weight / max(
            sum(ws.window.duration for ws in window_scores), 1e-9
        )
        if coverage < 0.10:
            warnings.append(
                f"only {coverage:.1%} of video duration qualified for scoring "
                f"({len(qualifying)}/{len(window_scores)} windows) — the score rests "
                "on a small slice of the video and should be treated as low-confidence"
            )

        return PrincipleResult(
            principle=self.name,
            score=round(score, 2),
            status="OK",
            na_reason=None,
            window_scores=tuple(window_scores),
            n_windows_total=len(window_scores),
            n_windows_qualifying=len(qualifying),
            params=self.params(),
            warnings=tuple(warnings),
        )

    def _unmeasurable_reason(self, window_scores: Sequence[WindowScore]) -> str | None:
        """Why the video could not be assessed at all, or None if it was.

        Returning None means "assessed, and no redundancy found", which the
        caller reports as 0. Returning a string means "not assessed", reported
        as N/A. Subclasses override this because only they know which exclusion
        reasons represent a genuine finding and which represent a failure.
        """
        if not window_scores:
            return "no windows were produced (zero-duration or unreadable video)"
        return None

    def _exclusion_breakdown(self, window_scores: Sequence[WindowScore]) -> str:
        counts: dict[str, int] = {}
        for ws in window_scores:
            key = ws.exclusion_reason or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
