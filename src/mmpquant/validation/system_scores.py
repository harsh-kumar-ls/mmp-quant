"""The system's scores, read from each run's result.json.

Only the pre-registered variants are read -- fixed before any manual score
was seen, so the comparison cannot drift towards whatever happens to fit:

  redundancy  whole_video     reported score; a window with nothing to assess
                              counts 0 (decision 08 Oct 2026)        [primary]
              assessed_only   mean over assessed windows only (`assessed_score`);
                              a video or segment with nothing assessable is 0,
                              consistent with the text-free = 0 ruling
  coherence   music_only      reported score, weights 1.0 / 0.0     [primary]
              music_plus_off_topic   the approved 0.5 / 0.5 (`coherence.alternative`)

A score of None (N/A: no narration anywhere) is excluded from validation and
listed, never treated as 0.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .ground_truth import KEY_SEGMENT, KEY_VIDEO


@dataclass(frozen=True)
class Variant:
    name: str
    video: Callable[[dict], float | None]
    segment: Callable[[dict, dict], float | None]
    primary: bool = False


def _or_zero(value, ok: bool):
    return (value if value is not None else 0.0) if ok else None


def _red_block(p):
    return p


def _coh_block(p):
    return p.get("coherence")


def _alt_segment(p, s):
    coh = p["coherence"]
    alt = coh.get("alternative") or {}
    w = alt.get("weights")
    parts = s.get("parts") or {}
    if s.get("score") is None or not w:
        return None
    return round(100.0 * sum(float(w[k]) * (parts.get(k) or 0.0) for k in w), 2)


VARIANTS: dict[str, list[Variant]] = {
    "redundancy": [
        Variant("whole_video",
                video=lambda p: p["result"]["score"],
                segment=lambda p, s: s["score"], primary=True),
        Variant("assessed_only",
                video=lambda p: _or_zero(p.get("assessed_score"), p["result"]["score"] is not None),
                segment=lambda p, s: _or_zero(s.get("assessed_score"), s["score"] is not None)),
    ],
    "coherence": [
        Variant("music_only",
                video=lambda p: p["coherence"]["result"]["score"],
                segment=lambda p, s: s["score"], primary=True),
        Variant("music_plus_off_topic",
                video=lambda p: (p["coherence"].get("alternative") or {}).get("score"),
                segment=_alt_segment),
    ],
}


@dataclass
class SystemScores:
    table: pd.DataFrame  # keys + one column per variant
    variants: list[str]
    not_scored: dict[str, str] = field(default_factory=dict)  # video_id -> why
    config_mismatch: list[str] = field(default_factory=list)


def _segments(payload: dict, principle: str) -> list[dict]:
    if principle == "redundancy":
        return payload.get("segments", [])
    return (payload.get("coherence") or {}).get("segments", [])


def load_system_scores(
    runs_dir: Path,
    principle: str,
    level: str,
    video_ids: list[str],
    current_config: dict[str, Any] | None = None,
) -> SystemScores:
    variants = VARIANTS[principle]
    rows, not_scored, mismatch = [], {}, []
    for vid in sorted(set(video_ids)):
        path = Path(runs_dir) / vid / "result.json"
        if not path.exists():
            not_scored[vid] = "no run (score the video first)"
            continue
        payload = json.loads(path.read_text())
        if principle == "coherence" and not payload.get("coherence"):
            not_scored[vid] = "run has no coherence score (re-score it)"
            continue
        if current_config is not None and payload.get("config") != current_config:
            mismatch.append(vid)
        if level == "video":
            rows.append({KEY_VIDEO: vid, **{v.name: v.video(payload) for v in variants}})
        else:
            for s in _segments(payload, principle):
                rows.append({KEY_VIDEO: vid, KEY_SEGMENT: int(s["index"]),
                             **{v.name: v.segment(payload, s) for v in variants}})
    keys = [KEY_VIDEO] if level == "video" else [KEY_VIDEO, KEY_SEGMENT]
    table = pd.DataFrame(rows, columns=keys + [v.name for v in variants])
    return SystemScores(table=table, variants=[v.name for v in variants],
                        not_scored=not_scored, config_mismatch=mismatch)


def primary_variant(principle: str) -> str:
    return next(v.name for v in VARIANTS[principle] if v.primary)
