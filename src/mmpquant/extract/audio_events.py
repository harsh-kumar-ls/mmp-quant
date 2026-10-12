"""Sound-event detection: what is audible, second by second.

Feeds the coherence scorer's background-music component. Mayer's coherence
principle (background music and sounds) concerns music or sound effects added
underneath narration: the learner's auditory channel has to separate the voice
from the music. To measure that we need, for every second of the video, how
likely it is that speech is audible and how likely it is that music is.

Model
-----
The Audio Spectrogram Transformer fine-tuned on AudioSet
(`MIT/ast-finetuned-audioset-10-10-0.4593`): Google's 527-class sound ontology,
multi-label, so "Speech" and "Music" are scored independently and both can be
high at once -- exactly the speech-over-music case. A boring, widely cited
choice; no training or fine-tuning here.

Framing
-------
Frame i covers [i·hop, (i+1)·hop). It is classified from a `clip_s`-long clip
centred on the frame, zero-padded at the ends of the file, so each frame gets
some acoustic context (a 1 s clip is too short for the model to recognise
music reliably) while the timeline keeps 1 s resolution.

Everything the scorer uses is written to `audio_events.json`: the tracked
classes' probabilities per frame, plus the top classes per frame so a reader
can see *what* the model heard (e.g. "Music, Theme music, Speech").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SPEECH = "Speech"


@dataclass(frozen=True)
class AudioFrame:
    start: float
    end: float
    # probability per tracked AudioSet class, e.g. {"Speech": 0.82, "Music": 0.04}
    probs: dict[str, float]
    # the model's top classes for this frame, for the audit trail
    top: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "probs": self.probs,
            "top": [[label, p] for label, p in self.top],
        }


@dataclass(frozen=True)
class AudioEvents:
    frames: tuple[AudioFrame, ...]
    hop_s: float
    clip_s: float
    model: str
    tracked: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hop_s": self.hop_s,
            "clip_s": self.clip_s,
            "tracked": list(self.tracked),
            "n_frames": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AudioEvents:
        return cls(
            frames=tuple(
                AudioFrame(
                    start=f["start"],
                    end=f["end"],
                    probs=dict(f["probs"]),
                    top=tuple((label, p) for label, p in f.get("top", [])),
                )
                for f in payload.get("frames", [])
            ),
            hop_s=float(payload.get("hop_s", 1.0)),
            clip_s=float(payload.get("clip_s", 3.0)),
            model=str(payload.get("model", "")),
            tracked=tuple(payload.get("tracked", [])),
        )


def frame_clips(n_samples: int, sr: int, hop_s: float, clip_s: float):
    """(frame_start_s, frame_end_s, clip_lo, clip_hi) for every frame.

    `clip_lo`/`clip_hi` are sample indices and may fall outside the signal;
    the caller zero-pads. A final partial frame is kept (it is real audio).
    """
    hop = round(hop_s * sr)
    clip = round(clip_s * sr)
    out = []
    i = 0
    while i * hop < n_samples:
        f_lo = i * hop
        f_hi = min((i + 1) * hop, n_samples)
        centre = (f_lo + f_hi) // 2
        out.append((f_lo / sr, f_hi / sr, centre - clip // 2, centre - clip // 2 + clip))
        i += 1
    return out


def _padded(x, lo: int, hi: int):
    import numpy as np

    seg = x[max(lo, 0):min(hi, len(x))]
    pad_left = max(0, -lo)
    pad_right = max(0, hi - len(x))
    if pad_left or pad_right:
        seg = np.concatenate([np.zeros(pad_left, x.dtype), seg, np.zeros(pad_right, x.dtype)])
    return seg


def read_wav_mono(path: Path):
    """16-bit / float WAV -> float32 mono in [-1, 1], plus sample rate."""
    import numpy as np
    from scipy.io import wavfile

    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def detect(
    wav_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    clip_s: float = 3.0,
    hop_s: float = 1.0,
    track: Sequence[str] = (SPEECH, "Music"),
    top_k: int = 5,
    device: str = "cpu",
    batch_size: int = 8,
) -> AudioEvents:  # pragma: no cover - requires model download
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    x, sr = read_wav_mono(wav_path)
    extractor = ASTFeatureExtractor.from_pretrained(model_name)
    if sr != extractor.sampling_rate:
        raise ValueError(
            f"{wav_path}: {sr} Hz audio, model expects {extractor.sampling_rate} Hz"
        )
    model = ASTForAudioClassification.from_pretrained(model_name).eval().to(device)
    labels = model.config.id2label
    index = {v: k for k, v in labels.items()}
    missing = [t for t in track if t not in index]
    if missing:
        raise ValueError(f"not AudioSet classes of {model_name}: {missing}")

    spans = frame_clips(len(x), sr, hop_s, clip_s)
    frames: list[AudioFrame] = []
    for b in range(0, len(spans), batch_size):
        batch = spans[b:b + batch_size]
        clips = [_padded(x, lo, hi) for _, _, lo, hi in batch]
        feats = extractor(clips, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(**feats).logits).cpu()
        for (start, end, _, _), p in zip(batch, probs, strict=True):
            top_idx = p.argsort(descending=True)[:top_k].tolist()
            frames.append(
                AudioFrame(
                    start=round(start, 3),
                    end=round(end, 3),
                    probs={t: round(float(p[index[t]]), 4) for t in track},
                    top=tuple((labels[i], round(float(p[i]), 4)) for i in top_idx),
                )
            )
    log.info("audio events: %d frames from %s", len(frames), wav_path)
    return AudioEvents(
        frames=tuple(frames), hop_s=hop_s, clip_s=clip_s, model=model_name,
        tracked=tuple(track),
    )
