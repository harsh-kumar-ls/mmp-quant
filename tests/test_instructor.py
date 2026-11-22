"""Instructor score = 100 - (R + C) / 2 (user, 12 Nov 2026)."""

from __future__ import annotations

import pytest

from mmpquant.instructor import FORMULA, instructor_block, instructor_score


@pytest.mark.parametrize("r, c, expected", [
    (0, 0, 100.0),          # best possible design
    (100, 100, 0.0),        # worst possible
    (70, 90, 20.0),         # the user's first formula gave -60 here
    (26.03, 98.63, 37.67),
])
def test_formula(r, c, expected):
    assert instructor_score(r, c) == pytest.approx(expected)


@pytest.mark.parametrize("r, c", [(None, 50.0), (50.0, None), (None, None)])
def test_na_when_either_score_is_na(r, c):
    assert instructor_score(r, c) is None


def _payload(red, coh, red_segs, coh_segs):
    return {
        "result": {"score": red},
        "segments": [{"index": i, "timecode": f"t{i}", "score": s} for i, s in enumerate(red_segs)],
        "coherence": {"result": {"score": coh},
                      "segments": [{"index": i, "score": s} for i, s in enumerate(coh_segs)]},
    }


def test_block_per_video_and_per_segment():
    b = instructor_block(_payload(40.0, 60.0, [20.0, None], [80.0, 40.0]))
    assert b["formula"] == FORMULA and b["score"] == 50.0
    assert [s["score"] for s in b["segments"]] == [50.0, None]


def test_no_block_without_coherence():
    assert instructor_block({"result": {"score": 10.0}, "segments": [], "coherence": None}) is None
