"""`_apply_video_score`: where the reported video score replaces the scorer's.

The scorer's `result.score` is the mean over assessed windows only. Since
08 Oct 2026 the reported score is the whole-video mean in which an unassessed
window counts 0 (`segments.aggregate_video`, tested in test_segments.py, is the
arithmetic). This is the wiring around it: swapping the number into a
`PrincipleResult`, leaving N/A alone, and flagging a large gap between the two
figures rather than silently reporting one.
"""

from __future__ import annotations

import pytest

from mmpquant.pipeline import DIVERGENT_AGGREGATION_THRESHOLD, _apply_video_score
from mmpquant.segments import VideoAggregate
from mmpquant.types import PrincipleResult


def result(score, *, status="OK", warnings=()):
    return PrincipleResult(
        principle="redundancy", score=score, status=status, na_reason=None,
        window_scores=(), n_windows_total=1, n_windows_qualifying=1 if score else 0,
        params={}, warnings=tuple(warnings),
    )


# Test_13's real figures: 2 of 18 windows assessed, 5.67 where assessed.
TEST_13 = VideoAggregate(score=1.58, assessed_score=5.67, coverage=0.2778)
COVERED = VideoAggregate(score=40.0, assessed_score=40.0, coverage=1.0)


class TestScoreIsReplaced:
    def test_ok_result_gets_the_whole_video_score(self):
        new = _apply_video_score(result(5.67), TEST_13, video_label="Test_13")
        assert new.score == 1.58

    def test_na_result_is_returned_unchanged(self):
        na_result = result(None, status="N/A")
        assert _apply_video_score(na_result, TEST_13, video_label="demo") is na_result

    def test_text_free_video_scores_zero(self):
        # User decision 08 Oct 2026: nothing on screen anywhere -> 0, not N/A.
        agg = VideoAggregate(score=0.0, assessed_score=None, coverage=0.0)
        new = _apply_video_score(result(0.0), agg, video_label="demo")
        assert new.score == 0.0 and new.status == "OK"
        assert new.warnings == ()


class TestDivergenceWarning:
    def test_fully_covered_video_is_silent(self):
        new = _apply_video_score(result(40.0), COVERED, video_label="demo")
        assert new.warnings == ()

    def test_low_coverage_names_both_numbers_and_the_coverage(self):
        # Gap 5.67 - 1.58 = 4.09 > 3.0.
        new = _apply_video_score(result(5.67), TEST_13, video_label="Test_13")
        warning = next(w for w in new.warnings if "on-screen sentence to assess" in w)
        assert "5.67" in warning and "1.58" in warning and "28%" in warning

    def test_gap_at_or_below_threshold_is_silent(self):
        agg = VideoAggregate(score=17.0, assessed_score=20.0, coverage=0.85)
        assert _apply_video_score(result(20.0), agg, video_label="demo").warnings == ()

    def test_threshold_is_a_named_constant_not_a_magic_number(self):
        assert DIVERGENT_AGGREGATION_THRESHOLD == 3.0

    def test_warning_appends_to_existing_warnings_not_replacing_them(self):
        new = _apply_video_score(
            result(5.67, warnings=["pre-existing warning"]), TEST_13, video_label="demo"
        )
        assert "pre-existing warning" in new.warnings
        assert len(new.warnings) == 2


class TestImpossibleCaseIsGuardedNotSilent:
    def test_ok_status_with_no_measured_time_raises(self):
        # status="OK" with score=None would violate the invariant every other
        # part of the project relies on (score is None iff N/A): fail loudly.
        agg = VideoAggregate(score=None, assessed_score=None, coverage=0.0)
        with pytest.raises(AssertionError, match="status is OK"):
            _apply_video_score(result(50.0), agg, video_label="demo")
