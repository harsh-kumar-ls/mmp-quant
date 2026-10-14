"""Coherence principle scorer (approved by the user, 11 Oct 2026).

Mayer's coherence principle: people learn better when extraneous material is
excluded. Two of its three forms are measured here; the third (decorative
pictures and unneeded detail) is out of scope -- see PROGRESS.md for why.

Score range 0-100. Higher = more extraneous material = worse, the same
direction as redundancy.

Per 10 s window (the same windows as redundancy):

    A  background music  = share of the window's SPOKEN seconds during which a
                           tracked sound class (default: AudioSet "Music") has
                           probability >= threshold.
                           Music with nobody speaking (an intro jingle) does
                           not count: Mayer's effect is music competing with
                           narration for the auditory channel. It is still
                           recorded as evidence.

    B  off-topic         = 1 if the window's narration is nearer, by meaning,
                           to unrelated everyday text than to the video's
                           typical narration; else 0.

                           r      = cos(window, centroid of the OTHER windows'
                                    narration)
                           typical = median r over the video's windows
                           unrelated = median cos(reference text, centroid)
                           off-topic  iff  r < unrelated + d·(typical - unrelated)
                                          (d = decision_point, 0.5 = midpoint)

                           Windows with fewer than `min_narration_words`
                           spoken words are not judged (B = 0, recorded).

    window score = 100 × (w_music·A + w_off·B)

A window qualifies when someone speaks in it. Segment and video scores are
formed by `mmpquant.segments` exactly as for redundancy: a duration-weighted
mean over all windows, an unqualified window counting 0. No narration
anywhere -> N/A: neither part can be judged without speech.

Known limitations, stated in the report: the off-topic test is relative to the
video itself, so a video that is off-topic throughout cannot be detected, and
a legitimate change of subtopic can look like a digression.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from ..textnorm import word_count
from ..types import PrincipleResult, Window, WindowScore
from .base import PrincipleScorer, VideoContext, register

EXCL_NO_NARRATION = "no_narration"

PART_MUSIC = "background_music"
PART_OFF_TOPIC = "off_topic"


def load_reference_texts(name_or_path: str) -> tuple[str, ...]:
    """Paragraphs from a reference file: package data name, or a path."""
    path = Path(name_or_path)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = resources.files("mmpquant.data").joinpath(name_or_path).read_text(
            encoding="utf-8")
    return tuple(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _norm(v):
    import numpy as np

    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@register
class CoherenceScorer(PrincipleScorer):
    name = "coherence"

    def __init__(
        self,
        encoder,
        *,
        weight_music: float = 0.5,
        weight_off_topic: float = 0.5,
        music_classes: Sequence[str] = ("Music",),
        music_threshold: float = 0.3,
        min_narration_words: int = 8,
        decision_point: float = 0.5,
        min_assessed_windows: int = 3,
        reference_texts: Sequence[str] = (),
        reference_name: str = "",
        embedding_model_name: str = "",
        audio_model_name: str = "",
    ) -> None:
        self.encoder = encoder
        self.weight_music = weight_music
        self.weight_off_topic = weight_off_topic
        self.music_classes = tuple(music_classes)
        self.music_threshold = music_threshold
        self.min_narration_words = min_narration_words
        self.decision_point = decision_point
        self.min_assessed_windows = min_assessed_windows
        self.reference_texts = tuple(reference_texts)
        self.reference_name = reference_name
        self.embedding_model_name = embedding_model_name
        self.audio_model_name = audio_model_name

    @classmethod
    def from_config(cls, cfg, encoder) -> CoherenceScorer:
        c = "scoring.coherence."
        ref = str(cfg.get(c + "off_topic.reference_texts"))
        return cls(
            encoder,
            weight_music=float(cfg.get(c + "weights.background_music")),
            weight_off_topic=float(cfg.get(c + "weights.off_topic")),
            music_classes=tuple(cfg.get(c + "background_music.sound_classes")),
            music_threshold=float(cfg.get(c + "background_music.threshold")),
            min_narration_words=int(cfg.get(c + "off_topic.min_narration_words")),
            decision_point=float(cfg.get(c + "off_topic.decision_point")),
            min_assessed_windows=int(cfg.get(c + "off_topic.min_assessed_windows")),
            reference_texts=load_reference_texts(ref),
            reference_name=ref,
            embedding_model_name=str(cfg.get(c + "off_topic.embedding_model")),
            audio_model_name=str(cfg.get(c + "background_music.model")),
        )

    def params(self) -> dict[str, Any]:
        return {
            "weights": {PART_MUSIC: self.weight_music, PART_OFF_TOPIC: self.weight_off_topic},
            "music_classes": list(self.music_classes),
            "music_threshold": self.music_threshold,
            "audio_model": self.audio_model_name,
            "min_narration_words": self.min_narration_words,
            "decision_point": self.decision_point,
            "min_assessed_windows": self.min_assessed_windows,
            "reference_texts": self.reference_name,
            "n_reference_texts": len(self.reference_texts),
            "embedding_model": self.embedding_model_name,
        }

    # ------------------------------------------------------------------ #

    def score_window(self, window: Window) -> WindowScore:
        """Not meaningful alone: both parts need the whole video's context
        (the sound timeline, and the video's typical narration)."""
        raise NotImplementedError("CoherenceScorer scores a whole video: use score()")

    def score(
        self,
        windows: Sequence[Window],
        extra_warnings=(),
        context: VideoContext | None = None,
    ) -> PrincipleResult:
        if context is None or context.transcript is None or context.audio_events is None:
            raise ValueError("coherence needs the transcript and audio events (VideoContext)")

        words = context.transcript.words
        frames = context.audio_events.frames
        warnings = list(extra_warnings)

        per_window = []
        for w in windows:
            w_words = [x for x in words if w.start <= x.midpoint < w.end]
            narration = " ".join(x.text.strip() for x in w_words).strip()
            music = self._music(w, words, frames)
            per_window.append((w, narration, music))

        off = self._off_topic([narration for _, narration, _ in per_window])
        if off["note"]:
            warnings.append(off["note"])

        scores = []
        for i, (w, narration, music) in enumerate(per_window):
            judged = off["judged"][i]
            is_off = bool(off["off_topic"][i]) if judged else False
            off_evidence = {
                "judged": judged,
                "off_topic": is_off if judged else None,
                "relevance": off["relevance"][i],
                "typical": off["typical"],
                "unrelated": off["unrelated"],
                "cut": off["cut"],
                "n_words": word_count(narration),
                "reason_not_judged": None if judged else off["why_not"][i],
            }
            evidence = {"narration": narration, "music": music, PART_OFF_TOPIC: off_evidence}
            if music["narrated_s"] <= 0:
                scores.append(WindowScore(
                    window=w, score=None, qualifies=False,
                    components={PART_MUSIC: 0.0, PART_OFF_TOPIC: 0.0,
                                "narrated_s": 0.0, "music_s": 0.0},
                    exclusion_reason=EXCL_NO_NARRATION, evidence=evidence,
                ))
                continue
            a = music["share"]
            b = 1.0 if is_off else 0.0
            scores.append(WindowScore(
                window=w,
                score=round(100.0 * (self.weight_music * a + self.weight_off_topic * b), 4),
                qualifies=True,
                components={
                    PART_MUSIC: round(a, 4),
                    PART_OFF_TOPIC: b,
                    "narrated_s": music["narrated_s"],
                    "music_s": music["music_under_speech_s"],
                },
                evidence=evidence,
            ))

        return self.aggregate(tuple(scores), extra_warnings=warnings)

    # ------------------------------------------------------------------ #

    def _music(self, w: Window, words, frames) -> dict[str, Any]:
        """Background-music share of the spoken seconds in one window."""
        narrated = music_under = music_alone = 0.0
        music_times: list[float] = []
        jingle_times: list[float] = []
        spoken_times: list[float] = []
        level_diffs: list[float] = []
        labels: dict[str, int] = {}
        for f in frames:
            mid = (f.start + f.end) / 2.0
            if not (w.start <= mid < w.end):
                continue
            dur = f.end - f.start
            spoken = any(x.start < f.end and x.end > f.start for x in words)
            has_music = any(f.probs.get(c, 0.0) >= self.music_threshold
                            for c in self.music_classes)
            if spoken:
                narrated += dur
                spoken_times.append(f.start)
                if has_music:
                    music_under += dur
                    music_times.append(f.start)
                    if f.level_db is not None and f.floor_db is not None:
                        level_diffs.append(round(f.floor_db - f.level_db, 1))
                    for label, _ in f.top[:3]:
                        labels[label] = labels.get(label, 0) + 1
            elif has_music:
                music_alone += dur
                jingle_times.append(f.start)
        return {
            "narrated_s": round(narrated, 3),
            "music_under_speech_s": round(music_under, 3),
            "music_without_speech_s": round(music_alone, 3),
            "share": (music_under / narrated) if narrated > 0 else 0.0,
            "spoken_at": spoken_times,
            "music_under_speech_at": music_times,
            "music_without_speech_at": jingle_times,
            "heard": [k for k, _ in sorted(labels.items(), key=lambda kv: -kv[1])[:5]],
            # per second with music under speech: quietest 100 ms minus the
            # second's level (dB). Evidence only (user decision, 14 Oct 2026).
            "music_to_speech_db": level_diffs,
        }

    def _off_topic(self, narrations: Sequence[str]) -> dict[str, Any]:
        """Judge every window's narration against the video's own topic."""
        import numpy as np

        n = len(narrations)
        out: dict[str, Any] = {
            "judged": [False] * n, "off_topic": [False] * n, "relevance": [None] * n,
            "why_not": [None] * n, "typical": None, "unrelated": None, "cut": None,
            "note": None,
        }
        counts = [word_count(t) for t in narrations]
        idx = [i for i, c in enumerate(counts) if c >= self.min_narration_words]
        for i in range(n):
            if counts[i] < self.min_narration_words:
                out["why_not"][i] = (
                    f"{counts[i]} spoken word(s); at least {self.min_narration_words} "
                    "are needed to judge its topic")
        if len(idx) < max(self.min_assessed_windows, 2):
            out["note"] = (
                f"off-topic part not assessed: only {len(idx)} window(s) have "
                f">= {self.min_narration_words} spoken words (need "
                f"{self.min_assessed_windows}) — too little narration to establish the "
                "video's typical topic")
            for i in idx:
                out["why_not"][i] = "too little narration in the video to judge topic"
            return out
        if not self.reference_texts:
            raise ValueError("off-topic test needs reference texts")

        emb = np.asarray(self.encoder.encode([narrations[i] for i in idx]), dtype=float)
        weights = np.asarray([counts[i] for i in idx], dtype=float)
        total = (emb * weights[:, None]).sum(axis=0)
        centroid = _norm(total)

        relevance = []
        for k, _ in enumerate(idx):
            others = _norm(total - weights[k] * emb[k])
            relevance.append(float(emb[k] @ others))
        typical = statistics.median(relevance)

        ref = np.asarray(self.encoder.encode(list(self.reference_texts)), dtype=float)
        unrelated = statistics.median(float(r @ centroid) for r in ref)

        out["typical"] = round(typical, 4)
        out["unrelated"] = round(unrelated, 4)
        if typical - unrelated <= 0:
            out["note"] = (
                "off-topic part not assessed: the video's narration is no closer to "
                f"itself (typical {typical:.3f}) than to unrelated text ({unrelated:.3f}), "
                "so there is no topic to drift away from")
            for i in idx:
                out["why_not"][i] = "no distinguishable topic in this video's narration"
            return out
        cut = unrelated + self.decision_point * (typical - unrelated)
        out["cut"] = round(cut, 4)
        for k, i in enumerate(idx):
            out["judged"][i] = True
            out["relevance"][i] = round(relevance[k], 4)
            out["off_topic"][i] = relevance[k] < cut
        return out

    def _unmeasurable_reason(self, window_scores) -> str | None:
        base = super()._unmeasurable_reason(window_scores)
        if base is not None:
            return base
        if all(ws.exclusion_reason == EXCL_NO_NARRATION for ws in window_scores):
            return (
                f"no narration detected in any of {len(window_scores)} windows — "
                "neither background music under speech nor off-topic narration can "
                "be judged without speech."
            )
        return None


def music_loudness_db(window_scores: Sequence[WindowScore]) -> dict[str, float | int | None]:
    """Median music level relative to speech over every second with music
    under speech (dB; nearer 0 = music nearly as loud as the voice).

    Evidence only, not part of any score (user decision, 14 Oct 2026). It
    separated test_7 (-5.8 dB) from test6 / Test_11 (about -13 dB), matching
    the user's listening -- but the measure was designed after hearing that,
    so it is suggestive, not validated.
    """
    values = [v for ws in window_scores
              for v in ((ws.evidence or {}).get("music") or {}).get("music_to_speech_db", [])]
    if not values:
        return {"median_db": None, "n_seconds": 0}
    return {"median_db": round(statistics.median(values), 1), "n_seconds": len(values)}


def part_scores(window_scores: Sequence[WindowScore]) -> dict[str, float | None]:
    """Each part on its own 0-100 scale, by the whole-video averaging rule.

    Same denominator as the headline score (all windows), so
    score = w_music·background_music + w_off·off_topic exactly.
    """
    measured = sum(ws.window.duration for ws in window_scores)
    if measured <= 0:
        return {PART_MUSIC: None, PART_OFF_TOPIC: None}
    out: dict[str, float | None] = {}
    for part in (PART_MUSIC, PART_OFF_TOPIC):
        total = sum(ws.components.get(part, 0.0) * ws.window.duration
                    for ws in window_scores if ws.qualifies)
        out[part] = round(100.0 * total / measured, 2)
    return out
