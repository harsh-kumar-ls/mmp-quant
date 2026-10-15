"""Stage orchestration.

Each extraction stage runs in its own OS process and communicates only through
JSON artifacts on disk. Two reasons, both practical:

  1. faster-whisper (PyAV) and RapidOCR (OpenCV) each bundle a copy of
     `libavdevice`. Loaded into one process on macOS, the ObjC runtime warns
     about duplicate class registration and threatens "spurious casting failures
     and mysterious crashes". Isolation removes that class of failure outright.
  2. Every stage becomes independently runnable, resumable and inspectable. A
     bad score can be traced by re-running one stage, and re-scoring with new
     config parameters does not re-run a 20-minute transcription.

Invoked as a module for the child processes:
    python -m mmpquant.stages asr <video> <outdir> <config>
    python -m mmpquant.stages ocr <video> <outdir> <config>
    python -m mmpquant.stages audio <video> <outdir> <config>
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .extract import media
from .extract.align import (
    OnscreenParams,
    WindowingParams,
    blocks_from_groups,
    build_windows,
)
from .extract.ocr.base import assess_reading_quality
from .types import AsrSegment, AsrWord, OcrBlock, OcrLine, Transcript, Window

log = logging.getLogger(__name__)

TRANSCRIPT_JSON = "transcript.json"
OCR_JSON = "ocr.json"
AUDIO_EVENTS_JSON = "audio_events.json"
WINDOWS_JSON = "windows.json"


class StageError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# ASR stage
# --------------------------------------------------------------------------- #


def run_asr_stage(video: Path, outdir: Path, cfg: Config) -> dict:
    """Extract audio, transcribe, write transcript.json. Runs in a child process."""
    from .extract import asr

    outdir.mkdir(parents=True, exist_ok=True)
    duration = media.duration_seconds(video)

    if not media.has_audio_stream(video):
        payload = Transcript((), None, None, duration).to_dict()
        payload["warnings"] = ["video has no audio stream — redundancy will be N/A"]
        _write_json(outdir / TRANSCRIPT_JSON, payload)
        return payload

    wav = media.extract_audio(
        video, outdir / "audio.wav", int(cfg.get("media.audio_sample_rate"))
    )
    transcript = asr.transcribe(
        wav,
        model_name=str(cfg.get("asr.model")),
        device=str(cfg.get("asr.device")),
        compute_type=str(cfg.get("asr.compute_type")),
        beam_size=int(cfg.get("asr.beam_size")),
        vad_filter=bool(cfg.get("asr.vad_filter")),
        language=cfg.get("asr.language", None),
        word_timestamps=bool(cfg.get("asr.word_timestamps")),
        duration_hint=duration,
    )

    payload = transcript.to_dict()
    payload["warnings"] = asr.language_warnings(
        transcript,
        cfg.get("asr.expected_language", None),
        float(cfg.get("asr.min_language_probability", 0.5)),
        decoded_as=cfg.get("asr.language", None),
    )
    if transcript.segments and not transcript.words:
        payload["warnings"].append(
            "no word-level timestamps returned — window assignment falls back to "
            "segment granularity and is coarser than the ±tolerance assumes"
        )
    _write_json(outdir / TRANSCRIPT_JSON, payload)
    return payload


# --------------------------------------------------------------------------- #
# OCR stage
# --------------------------------------------------------------------------- #


def run_ocr_stage(video: Path, outdir: Path, cfg: Config) -> dict:
    """Sample frames, dedup, OCR representatives, write ocr.json. Child process."""
    from .extract.ocr import (
        RELIABILITY_UNREADABLE,
        assess_recognition_reliability,
        dedup,
        get_backend,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    duration = media.duration_seconds(video)
    frames_dir = outdir / "frames"

    frames = media.sample_frames(
        video,
        frames_dir,
        fps=float(cfg.get("media.frame_fps")),
        image_format=str(cfg.get("media.frame_format", "png")),
    )

    width = int(cfg.get("ocr.dedup.downscale_width"))
    signatures = [dedup.frame_signature(f.path, width) for f in frames]
    groups = dedup.group_frames(
        signatures,
        threshold=float(cfg.get("ocr.dedup.threshold")),
        method=str(cfg.get("ocr.dedup.method", dedup.METHOD_CHANGED_PIXELS)),
        pixel_tolerance=float(cfg.get("ocr.dedup.pixel_tolerance", 0.1)),
    )
    ratio = dedup.dedup_ratio(len(frames), len(groups))
    log.info(
        "dedup: %d frames -> %d blocks (%.1f%% of OCR work avoided)",
        len(frames), len(groups), ratio * 100,
    )

    backend = get_backend(
        str(cfg.get("ocr.backend")),
        min_confidence=float(cfg.get("ocr.min_line_confidence")),
    )
    frame_results = [backend.read(frames[g.representative_index].path) for g in groups]
    lines_per_group = [fr.lines for fr in frame_results]

    blocks = blocks_from_groups(
        [f.timestamp for f in frames], groups, lines_per_group, duration
    )

    # Detection vs recognition, kept separate. See ocr/base.py: "text detected
    # but unreadable" must never be reported as "no text on screen".
    #
    # Blocks are 1:1 with groups and carry the timespan each one covers, so the
    # reliability check is weighted by duration. Without that, a static board
    # held for a minute counts once against a hundred frames of camera movement,
    # and the handwriting safeguard silently stops firing.
    reliability = assess_recognition_reliability(
        frame_results, weights=[b.end - b.start for b in blocks]
    )

    warnings: list[str] = []
    if ratio < 0.2 and len(frames) > 20:
        warnings.append(dedup_warning(ratio))
    if reliability["reliability"] == RELIABILITY_UNREADABLE:
        warnings.append("UNRELIABLE OCR: " + reliability["note"])
    elif all(not b.lines for b in blocks):
        warnings.append("no on-screen text detected anywhere — redundancy will be N/A")

    payload = {
        "n_frames_sampled": len(frames),
        "n_blocks": len(blocks),
        "dedup_ratio": round(ratio, 4),
        "backend": backend.describe(),
        "recognition_reliability": reliability,
        "warnings": warnings,
        "blocks": [b.to_dict() for b in blocks],
    }
    _write_json(outdir / OCR_JSON, payload)

    if not bool(cfg.get("output.keep_all_frames", False)):
        _prune_frames(frames, {g.representative_index for g in groups})

    return payload


def _prune_frames(frames: list[media.SampledFrame], keep: set[int]) -> None:
    """Delete non-representative frames.

    A 40-minute lecture at 2 fps is ~4800 PNGs — several gigabytes per run, which
    makes keeping every run's artifacts (a thesis requirement) impractical. The
    frames actually sent to OCR are kept, because those are the ones needed to
    audit a claim like "this text appeared at 14:32". The rest are reproducible
    from the video and the config at any time.
    """
    removed = 0
    for frame in frames:
        if frame.index not in keep and frame.path.exists():
            frame.path.unlink()
            removed += 1
    log.info("pruned %d non-representative frames (kept %d)", removed, len(keep))


# --------------------------------------------------------------------------- #
# Audio-events stage (coherence: background music)
# --------------------------------------------------------------------------- #


def run_audio_stage(video: Path, outdir: Path, cfg: Config) -> dict:
    """Classify the soundtrack second by second, write audio_events.json.

    Child process, like ASR and OCR: the audio model (~350 MB) is released
    when it exits instead of staying resident next to the parser and the
    embedding model -- this machine has 8 GB. Reads the audio.wav the ASR
    stage extracted, so it must run after ASR.
    """
    from .extract import audio_events

    outdir.mkdir(parents=True, exist_ok=True)
    wav = outdir / "audio.wav"
    prefix = "scoring.coherence.background_music"
    if not wav.exists():
        payload = audio_events.AudioEvents(
            (), float(cfg.get(f"{prefix}.hop_s")), float(cfg.get(f"{prefix}.clip_s")),
            str(cfg.get(f"{prefix}.model")), (),
        ).to_dict()
        payload["warnings"] = ["no audio track — background music cannot be assessed"]
        _write_json(outdir / AUDIO_EVENTS_JSON, payload)
        return payload

    events = audio_events.detect(
        wav,
        model_name=str(cfg.get(f"{prefix}.model")),
        clip_s=float(cfg.get(f"{prefix}.clip_s")),
        hop_s=float(cfg.get(f"{prefix}.hop_s")),
        track=(audio_events.SPEECH, *cfg.get(f"{prefix}.sound_classes")),
        device=str(cfg.get(f"{prefix}.device", "cpu")),
    )
    payload = events.to_dict()
    payload["warnings"] = []
    _write_json(outdir / AUDIO_EVENTS_JSON, payload)
    return payload


def load_audio_events(path: Path):
    from .extract.audio_events import AudioEvents

    payload = json.loads(path.read_text())
    return AudioEvents.from_dict(payload), list(payload.get("warnings", []))


def enabled_principles(cfg: Config) -> list[str]:
    return list(cfg.get("scoring.principles", ["redundancy"]))


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #


def load_transcript(path: Path) -> tuple[Transcript, list[str]]:
    payload = json.loads(path.read_text())
    segments = tuple(
        AsrSegment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=tuple(AsrWord(w["start"], w["end"], w["text"]) for w in s["words"]),
        )
        for s in payload["segments"]
    )
    transcript = Transcript(
        segments=segments,
        language=payload.get("language"),
        language_probability=payload.get("language_probability"),
        duration=float(payload.get("duration", 0.0)),
    )
    return transcript, list(payload.get("warnings", []))


DEDUP_WARNING_PREFIX = "frame deduplication was ineffective"


def dedup_warning(ratio: float) -> str:
    """The warning for a video whose frames could not be de-duplicated.

    Until 04 Oct 2026 this ended "This is a cost signal, not a correctness
    problem." That was wrong for the case it fires on most: a lecturer moving
    in front of a static board. Every frame then differs from the last, so
    every frame is OCR'd separately, each with a different part of the board
    hidden or blurred -- and a 10 s window collects twenty slightly different
    misreadings of the same text. On Test_22 that pile of variants is what let
    one garbled reading be tagged as a sentence and scored (defect 10.3).
    """
    return (
        f"{DEDUP_WARNING_PREFIX} ({ratio:.1%} of frames collapsed) — consecutive "
        "frames kept changing (a lecturer moving in front of the board, camera "
        "motion, animation, or a talking-head inset), so nearly every sampled frame "
        "was read by OCR separately. That costs time, and it can also affect the "
        "result: each frame is read independently with different parts of the text "
        "hidden or blurred, so one stretch of video can collect many slightly "
        "different misreadings of the same text. The OCR reliability check cannot "
        "always detect a misreading, so inspect a few frames before trusting the score."
    )


def load_ocr_reliability(path: Path) -> dict:
    """The recognition-reliability summary recorded by the OCR stage."""
    payload = json.loads(path.read_text())
    return payload.get("recognition_reliability", {})


def load_ocr(path: Path) -> tuple[tuple[OcrBlock, ...], list[str]]:
    payload = json.loads(path.read_text())
    blocks = tuple(
        OcrBlock(
            start=b["start"],
            end=b["end"],
            lines=tuple(
                OcrLine(line["text"], line["confidence"], tuple(line["bbox"]) if line.get("bbox") else None)
                for line in b["lines"]
            ),
            source_frame_time=b["source_frame_time"],
            n_frames_collapsed=b["n_frames_collapsed"],
        )
        for b in payload["blocks"]
    )
    # Runs cached before 04 Oct 2026 carry the old wording, which claimed this
    # was "not a correctness problem". Regenerate it from the recorded ratio
    # rather than leave a false statement in every report built from cache.
    warnings = [
        dedup_warning(float(payload.get("dedup_ratio", 0.0)))
        if w.startswith(DEDUP_WARNING_PREFIX)
        else w
        for w in payload.get("warnings", [])
    ]
    return blocks, warnings


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ExtractionResult:
    windows: list[Window]
    duration: float
    transcript: Transcript
    ocr_blocks: tuple[OcrBlock, ...]
    warnings: list[str]
    ocr_reliability: dict = field(default_factory=dict)
    # Only when coherence is enabled; None otherwise.
    audio_events: object | None = None
    audio_warnings: list[str] = field(default_factory=list)


def _spawn(stage: str, video: Path, outdir: Path, config_path: Path) -> None:
    cmd = [
        sys.executable, "-m", "mmpquant.stages",
        stage, str(video), str(outdir), str(config_path),
    ]
    log.info("spawning %s stage: %s", stage, " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise StageError(
            f"{stage} stage failed with exit code {proc.returncode}. "
            f"Its partial artifacts are in {outdir}."
        )


def run_extraction(
    video: Path,
    outdir: Path,
    cfg: Config,
    *,
    resume: bool = True,
    on_stage: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Run both extraction stages (as child processes) and build the windows.

    `resume=True` reuses an existing artifact instead of recomputing it. This is
    what makes re-scoring with different config weights cheap: only the
    alignment and scoring steps re-run, and they are seconds, not minutes.
    Note the trade-off — resume does NOT re-run ASR/OCR when *their* config
    changes, so a run directory is only valid for the media/asr/ocr settings
    that produced it. Use a fresh output directory when changing those.

    `on_stage`, if given, is called with a short stage name ("asr", "ocr",
    "aligning") as each one starts. Each ASR/OCR stage is a blocking child
    process that can run for minutes, so this is the only way a caller — the
    web UI in particular — can show the user which of the two long steps is
    currently running rather than a single opaque wait.
    """
    if cfg.source_path is None:
        raise StageError("config must be loaded from a file to run child stages")

    outdir.mkdir(parents=True, exist_ok=True)

    stages = [("asr", TRANSCRIPT_JSON), ("ocr", OCR_JSON)]
    if "coherence" in enabled_principles(cfg):
        stages.append(("audio", AUDIO_EVENTS_JSON))
    for stage, artifact in stages:
        target = outdir / artifact
        if resume and target.exists():
            log.info("reusing existing %s (resume=True)", artifact)
            continue
        if on_stage is not None:
            on_stage(stage)
        _spawn(stage, video, outdir, cfg.source_path)

    if on_stage is not None:
        on_stage("aligning")

    transcript, asr_warnings = load_transcript(outdir / TRANSCRIPT_JSON)
    ocr_blocks, ocr_warnings = load_ocr(outdir / OCR_JSON)
    duration = media.duration_seconds(video)

    params = WindowingParams(
        size_s=float(cfg.get("windowing.size_s")),
        tolerance_s=float(cfg.get("alignment.tolerance_s")),
        expand=str(cfg.get("alignment.expand")),
        keep_partial_final_window=bool(cfg.get("windowing.keep_partial_final_window")),
    )
    windows = build_windows(
        transcript,
        ocr_blocks,
        duration,
        params,
        onscreen=_onscreen_params(cfg),
    )

    _write_json(
        outdir / WINDOWS_JSON,
        {
            "duration": duration,
            "windowing": {
                "size_s": params.size_s,
                "tolerance_s": params.tolerance_s,
                "expand": params.expand,
                "asr_tolerance": params.asr_tolerance,
                "ocr_tolerance": params.ocr_tolerance,
            },
            "n_windows": len(windows),
            "windows": [w.to_dict() for w in windows],
        },
    )

    events, audio_warnings = (None, [])
    if (outdir / AUDIO_EVENTS_JSON).exists() and "coherence" in enabled_principles(cfg):
        events, audio_warnings = load_audio_events(outdir / AUDIO_EVENTS_JSON)
        wav = outdir / "audio.wav"
        if events.frames and events.frames[0].level_db is None and wav.exists():
            # Runs classified before 14 Oct 2026 lack loudness; it needs only
            # the WAV, not the model, so fill it in rather than re-classify.
            from .extract.audio_events import with_levels

            events = with_levels(events, wav)

    return ExtractionResult(
        windows=windows,
        duration=duration,
        transcript=transcript,
        ocr_blocks=ocr_blocks,
        warnings=asr_warnings + ocr_warnings,
        ocr_reliability=assess_reading_quality(
            load_ocr_reliability(outdir / OCR_JSON), ocr_blocks
        ),
        audio_events=events,
        audio_warnings=audio_warnings,
    )


def _onscreen_params(cfg: Config) -> OnscreenParams:
    """Every `onscreen:` setting from config.yaml -- see OnscreenParams for why
    these were previously documented but never read."""
    from .onscreen import set_spacy_model

    set_spacy_model(str(cfg.get("onscreen.spacy_model", "en_core_web_trf")))
    return OnscreenParams(
        min_prose_words=int(cfg.get("scoring.redundancy.min_onscreen_words")),
        line_gap_ratio=float(cfg.get("onscreen.line_gap_ratio", 1.6)),
        x_align_tol_px=float(cfg.get("onscreen.x_align_tol_px", 24.0)),
        dedup_similarity=float(cfg.get("onscreen.dedup_similarity", 0.88)),
        split_run_together_words=bool(cfg.get("onscreen.split_run_together_words", True)),
        caption_stitching=bool(cfg.get("onscreen.caption_stitching.enabled", True)),
        caption_max_gap_s=float(cfg.get("onscreen.caption_stitching.max_gap_s", 1.0)),
        caption_max_pieces=int(cfg.get("onscreen.caption_stitching.max_pieces", 4)),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("wrote %s", path)


def _main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if len(argv) != 4:
        print(__doc__)
        return 2
    stage, video, outdir, config_path = argv
    cfg = Config.load(config_path)
    runner = {"asr": run_asr_stage, "ocr": run_ocr_stage, "audio": run_audio_stage}[stage]
    runner(Path(video), Path(outdir), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
