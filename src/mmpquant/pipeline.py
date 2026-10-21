"""The score-one-video pipeline, shared by the CLI and the web UI.

Extracted from `cli.score()` so there is exactly one place that runs
extraction, scores it, aggregates segments and writes the artifacts. The CLI
and the web server differ only in how they report progress and present the
result — both call `run_pipeline` and get back the same payload that
`result.json` holds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Config
from .overlap import SentenceTransformerEmbedder
from .report import render_markdown
from .scoring.base import VideoContext
from .scoring.coherence import (
    PART_MUSIC,
    PART_OFF_TOPIC,
    CoherenceScorer,
    music_loudness_db,
    part_scores,
)
from .scoring.redundancy import RedundancyScorer
from .segments import (
    SegmentScore,
    VideoAggregate,
    aggregate_to_segments,
    aggregate_video,
    write_segment_scores,
)
from .stages import ExtractionResult, enabled_principles, run_extraction
from .types import PrincipleResult

# When the whole-video score and the assessed-only average differ by more than
# this many points, most of the video had nothing on screen to assess, and the
# warning says so with both numbers. A round number a few points above the
# rounding noise between the two on fully covered videos; flag, don't guess.
DIVERGENT_AGGREGATION_THRESHOLD = 3.0


@dataclass
class PipelineResult:
    payload: dict[str, Any]
    extraction: ExtractionResult
    result: PrincipleResult
    segments: list[SegmentScore]
    outdir: Path


def _apply_video_score(
    result: PrincipleResult,
    agg: VideoAggregate,
    *,
    video_label: str,
    assessable: str = "an on-screen sentence to assess",
) -> PrincipleResult:
    """Replace the scorer's assessed-only mean with the whole-video score.

    The scorer reports the mean over assessed windows (CLAUDE.md §2.6 as first
    written); the reported video score is the whole-video mean in which an
    unassessed window counts 0 (decision 08 Oct 2026 -- see
    `segments` "One averaging rule"). When the two differ by more than
    `DIVERGENT_AGGREGATION_THRESHOLD`, a warning names both and the coverage,
    rather than reporting one silently. A video that was N/A stays N/A.
    """
    if result.status != "OK":
        return result
    if agg.score is None:
        # score is None iff N/A: never let an OK result carry no number.
        raise AssertionError(
            f"{video_label}: status is OK but the windows have no duration"
        )

    extra: list[str] = []
    if (agg.assessed_score is not None
            and abs(agg.score - agg.assessed_score) > DIVERGENT_AGGREGATION_THRESHOLD):
        extra.append(
            f"only {agg.coverage:.0%} of the video had {assessable}: where it did, "
            f"{result.principle} averaged {agg.assessed_score:.2f}; "
            f"across the whole video (reported) it is {agg.score:.2f}. The score "
            f"rests on a small part of the video — check segment_scores.csv."
        )
    return replace(result, score=agg.score, warnings=result.warnings + tuple(extra))


def _score_coherence(
    extraction: ExtractionResult,
    cfg: Config,
    embedder: SentenceTransformerEmbedder,
    segment_length: float,
    video: Path,
    outdir: Path,
) -> dict[str, Any] | None:
    """Score coherence with the same windows, segments and averaging rule as
    redundancy, and write coherence_segment_scores.csv. None if not enabled."""
    if "coherence" not in enabled_principles(cfg):
        return None
    model = str(cfg.get("scoring.coherence.off_topic.embedding_model"))
    encoder = embedder if model == embedder.model_name else SentenceTransformerEmbedder(model)
    scorer = CoherenceScorer.from_config(cfg, encoder)
    result = scorer.score(
        extraction.windows,
        extra_warnings=extraction.audio_warnings,
        context=VideoContext(extraction.transcript, extraction.audio_events),
    )
    keys = (PART_MUSIC, PART_OFF_TOPIC)
    segments = aggregate_to_segments(
        result.window_scores, extraction.duration, segment_length, part_keys=keys
    )
    write_segment_scores(segments, video.stem, outdir / "coherence_segment_scores.csv",
                         part_keys=keys)
    alt = cfg.get("scoring.coherence.alternative_weights", None)
    return coherence_block(result, segments, video_label=str(video), alternative_weights=alt)


def coherence_block(
    result: PrincipleResult,
    segments: list[SegmentScore],
    *,
    video_label: str,
    alternative_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The `payload["coherence"]` dict: reported score, segments, and both
    parts, from a scored result. Pure, so tests build it the same way."""
    agg = aggregate_video(result.window_scores)
    result = _apply_video_score(result, agg, video_label=video_label,
                                assessable="narration to assess")
    ok = result.status == "OK"
    parts = part_scores(result.window_scores) if ok else None
    alternative = None
    if alternative_weights:
        # The pre-registered alternative weighting (config), from the same
        # parts: exact, since every score here is linear in the parts.
        alternative = {
            "weights": dict(alternative_weights),
            "score": round(sum(float(alternative_weights[k]) * parts[k] for k in parts), 2)
            if parts else None,
        }
    return {
        "result": result.to_dict(),
        "segments": [s.to_dict() for s in segments],
        "assessed_score": agg.assessed_score if ok else None,
        "coverage": agg.coverage,
        # each part on the headline's scale: score = Σ weight × part
        "parts": parts,
        "alternative": alternative,
        # evidence only, not in any score (user decision, 14 Oct 2026)
        "music_loudness": music_loudness_db(result.window_scores),
        "aggregation": "whole_video_mean",
    }


def run_pipeline(
    video: Path,
    outdir: Path,
    cfg: Config,
    *,
    no_resume: bool = False,
    on_stage: Callable[[str], None] | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
) -> PipelineResult:
    """Run extraction + scoring + segmenting, write result.json and report.md.

    Redundancy is always scored and stays at the top level of the payload
    (`result`, `segments`), so every earlier run and reader keeps working;
    coherence, when enabled, is under `payload["coherence"]`.

    `on_stage` is called with a short name ("asr", "ocr", "aligning",
    "scoring", "writing_report") as each step starts, so a long-running caller
    — the web UI — can show which stage a video is on rather than one opaque
    wait. It is never required: the CLI ignores it and prints its own output
    once the whole thing is done.
    """

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    extraction = run_extraction(video, outdir, cfg, resume=not no_resume, on_stage=stage)

    stage("scoring")
    # `embedder` lets a caller scoring many videos (the regression check)
    # load the sentence model once instead of once per video.
    if embedder is None or embedder.model_name != str(cfg.get("scoring.redundancy.embedding_model")):
        embedder = SentenceTransformerEmbedder(
            str(cfg.get("scoring.redundancy.embedding_model"))
        )
    scorer = RedundancyScorer.from_config(cfg, embedder)
    result = scorer.score(extraction.windows, extra_warnings=extraction.warnings,
                          context=VideoContext(extraction.transcript, None))

    segment_length = float(cfg.get("segments.length_s"))
    segments = aggregate_to_segments(
        result.window_scores, extraction.duration, segment_length
    )
    write_segment_scores(segments, video.stem, outdir / "segment_scores.csv")

    video_agg = aggregate_video(result.window_scores)
    result = _apply_video_score(result, video_agg, video_label=str(video))

    payload = {
        "video": str(video),
        "video_id": video.stem,
        "duration": extraction.duration,
        "segment_length_s": segment_length,
        "ocr_recognition_reliability": extraction.ocr_reliability,
        "segments": [s.to_dict() for s in segments],
        "result": result.to_dict(),
        # Mean over assessed windows only, and the share of time assessed:
        # score = assessed_score × coverage. Kept beside the reported score
        # as the pre-registered alternative for validation.
        "assessed_score": video_agg.assessed_score if result.status == "OK" else None,
        "coverage": video_agg.coverage,
        "aggregation": "whole_video_mean",
        # Second principle (11 Oct 2026); None when not enabled in config.
        "coherence": _score_coherence(extraction, cfg, embedder, segment_length, video, outdir),
        # The full config, verbatim, so any number here is traceable to the
        # exact parameters that produced it.
        "config": cfg.raw,
        "config_path": str(cfg.source_path),
    }

    stage("writing_report")
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (outdir / "report.md").write_text(render_markdown(payload), encoding="utf-8")

    return PipelineResult(
        payload=payload, extraction=extraction, result=result, segments=segments, outdir=outdir
    )
