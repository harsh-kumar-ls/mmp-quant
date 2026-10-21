"""The pre-registered rule for choosing the 10 videos the professor rates
(PROGRESS.md, 20 Oct 2026). Synthetic candidates only."""

from __future__ import annotations

import pytest

from mmpquant.selection import Candidate, select

STYLES = ["slides", "whiteboard", "pen", "animation"]


def cand(name, red, *, style=None, coh=0.0, dur=180.0):
    return Candidate(name, style or STYLES[hash(name) % 4], dur, red, coh, None)


def spread(n=20):
    """Candidates whose redundancy covers 0-95 in steps of 5, all four styles,
    every fourth one with music."""
    return [cand(f"v{i:02d}", i * 5.0, style=STYLES[i % 4], coh=50.0 if i % 4 == 1 else 0.0)
            for i in range(n)]


def test_one_video_per_target_and_nearest_scores():
    sel = select(spread())
    assert [t for t, _ in sel.slots] == list(range(0, 100, 10))
    assert len({c.video_id for _, c in sel.slots}) == 10
    assert sel.cost <= 10 * 5


def test_all_four_styles_and_music_constraints_are_met():
    sel = select(spread())
    chosen = [c for _, c in sel.slots]
    assert {c.style for c in chosen} == set(STYLES)
    assert sum(c.music for c in chosen) >= 3 and sum(not c.music for c in chosen) >= 3
    assert sel.unmet == []


def test_a_swap_never_trades_like_for_like():
    cands = spread()
    # remove most music so a swap is needed
    cands = [c if c.video_id in ("v01", "v05", "v09") else
             Candidate(c.video_id, c.style, c.duration_s, c.redundancy, 0.0, None) for c in cands]
    sel = select(cands)
    for line in sel.swaps:
        old, new = line.split(": ")[1].split(" (")[0].split(" -> ")
        assert old != new


def test_a_swap_costing_more_than_five_points_is_refused_and_reported():
    cands = [cand(f"r{i}", i * 10.0, style="slides") for i in range(10)]
    cands.append(cand("far_pen", 100.0, style="pen"))  # 10 points from the nearest target (90)
    sel = select(cands, min_music=0, min_no_music=0)
    assert "style:pen" in sel.unmet and all("far_pen" not in s for s in sel.swaps)


def test_ties_prefer_the_shorter_video_then_the_name():
    cands = [cand("b_long", 0.0, style="pen", dur=190), cand("a_short", 0.0, style="pen", dur=150)]
    cands += [cand(f"x{i}", i * 10.0, style=STYLES[i % 4]) for i in range(1, 10)]
    sel = select(cands, min_music=0, min_no_music=0)
    assert dict(sel.slots)[0].video_id == "a_short"


def test_too_few_candidates_is_an_error():
    with pytest.raises(ValueError, match="only 3 eligible"):
        select([cand(f"v{i}", i * 10.0) for i in range(3)])


def test_footage_is_eligible_but_does_not_count_as_one_of_the_four_styles():
    # 21 Oct 2026: filmed videos (presenter / stock footage with on-screen
    # text) are eligible -- the rule's eligibility is duration only -- but the
    # four styles must still each appear.
    cands = [cand(f"v{i:02d}", i * 5.0, style=STYLES[i % 4], coh=50.0 if i % 4 == 1 else 0.0)
             for i in range(19)]
    cands.append(cand("filmed_subtitled", 90.0, style="footage"))
    sel = select(cands)
    chosen = {c.video_id: c for _, c in sel.slots}
    assert "filmed_subtitled" in chosen            # nearest to target 90
    assert {c.style for c in chosen.values()} >= set(STYLES)


def test_amendment_ties_prefer_a_new_coherence_level():
    # 21 Oct 2026 amendment: test4 (0, coherence 100) vs a tie at 0 with
    # coherence 53 -> the tie goes to the video adding the 'mid' level.
    cands = [cand(f"v{i:02d}", i * 10.0, style=STYLES[i % 4], coh=0.0) for i in range(1, 10)]
    cands += [cand("music_a", 30.0, style="pen", coh=90.0), cand("music_b", 50.0, style="pen", coh=95.0)]
    cands += [cand("short_high", 0.0, style="animation", coh=100.0, dur=14),
              cand("mid_tie", 0.0, style="animation", coh=53.0, dur=180)]
    sel = select(cands, min_music=1, min_no_music=1)
    assert dict(sel.slots)[0].video_id == "mid_tie"
    assert any("amendment" in s for s in sel.swaps)
