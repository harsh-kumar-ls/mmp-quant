"""Instructor score: one number for how well a video follows both principles.

    instructor = 100 - (redundancy + coherence) / 2

Agreed with the user on 12 Nov 2026, after the professor asked for a final
"instructor score". Higher = better design (the opposite direction to the two
principle scores, where higher = worse). The mean, not the sum, keeps it on
0-100: the user's first proposal, 100 - (R + C), could fall to -100.

It adds no measurement of its own -- it is computed from the two reported
scores, per video and per segment, and is N/A whenever either one is N/A.
It is reported beside R and C, never instead of them: the professor's ratings
are compared with each principle separately, and the same formula applied to
her ratings gives her instructor score.
"""

from __future__ import annotations

from typing import Any

FORMULA = "100 - (redundancy + coherence) / 2"


def instructor_score(redundancy: float | None, coherence: float | None) -> float | None:
    if redundancy is None or coherence is None:
        return None
    return round(100.0 - (float(redundancy) + float(coherence)) / 2.0, 2)


def instructor_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    """`payload["instructor"]` from a result payload; None if coherence is off."""
    coh = payload.get("coherence")
    if not coh:
        return None
    red_score = (payload.get("result") or {}).get("score")
    coh_score = (coh.get("result") or {}).get("score")
    coh_segments = {s["index"]: s.get("score") for s in coh.get("segments", [])}
    return {
        "score": instructor_score(red_score, coh_score),
        "formula": FORMULA,
        "segments": [
            {"index": s["index"], "timecode": s.get("timecode"),
             "score": instructor_score(s.get("score"), coh_segments.get(s["index"]))}
            for s in payload.get("segments", [])
        ],
    }
