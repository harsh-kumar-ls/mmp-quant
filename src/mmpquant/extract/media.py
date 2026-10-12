"""ffmpeg wrappers: duration probe, audio extraction, frame sampling.

All media I/O goes through the ffmpeg/ffprobe CLI rather than a Python binding.
Reasons, in order of importance for this project:

  1. Every intermediate lands on disk as a real file (wav, png) that can be
     opened and inspected by hand — the thesis needs an auditable trail.
  2. The exact command line is logged, so a run is reproducible from the log
     alone on any machine with the same ffmpeg version.
  3. No native-library version coupling to the Python process.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp: float
    path: Path


def _binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"{name} not found on PATH. Install it (macOS: `brew install ffmpeg`)."
        )
    return found


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.info("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return proc


def probe(video: Path) -> dict:
    """Return the ffprobe JSON for a video, or raise MediaError."""
    if not video.exists():
        raise MediaError(f"video not found: {video}")
    proc = _run(
        [
            _binary("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video),
        ]
    )
    return json.loads(proc.stdout)


def duration_seconds(video: Path) -> float:
    info = probe(video)
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"could not determine duration of {video}") from exc


def has_audio_stream(video: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(video).get("streams", []))


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Decode to mono PCM WAV at `sample_rate` — what Whisper expects.

    Resampling here rather than inside the ASR library keeps the audio actually
    fed to the model on disk, so a bad transcript can be traced to bad audio.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _binary("ffmpeg"), "-nostdin", "-y",
            "-i", str(video),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )
    return out_wav


def sample_frames(
    video: Path,
    out_dir: Path,
    fps: float = 2.0,
    image_format: str = "png",
) -> list[SampledFrame]:
    """Sample frames at a constant `fps` into `out_dir`.

    Timestamp convention: the i-th emitted frame (0-based) is labelled i / fps.
    The `fps` filter selects the source frame nearest each target instant, so the
    true timestamp can differ by up to 1/(2*fps) — 0.25 s at the 2 fps default.
    That is an order of magnitude below the ±3 s alignment tolerance, so it does
    not affect scoring; it is recorded here because it bounds how precisely a
    reported "worst offending segment" timestamp can be trusted.

    PNG is the default: JPEG compression artifacts around small slide text
    measurably hurt OCR recall.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"frame_*.{image_format}"):
        stale.unlink()

    _run(
        [
            _binary("ffmpeg"), "-nostdin", "-y",
            "-i", str(video),
            # The fps filter already emits a constant-rate stream, so no
            # -vsync/-fps_mode flag is needed. Passing one is actively harmful:
            # ffmpeg 9 removed -vsync ("Option not found") while older builds
            # lack -fps_mode, so either spelling breaks on some version.
            "-vf", f"fps={fps}",
            str(out_dir / f"frame_%06d.{image_format}"),
        ]
    )

    paths = sorted(out_dir.glob(f"frame_*.{image_format}"))
    if not paths:
        raise MediaError(
            f"no frames extracted from {video} — no video stream, or an unsupported codec"
        )
    return [SampledFrame(index=i, timestamp=i / fps, path=p) for i, p in enumerate(paths)]
