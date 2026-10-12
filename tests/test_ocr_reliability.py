"""Detection vs recognition: "text present but unreadable" is not "no text".

Regression suite for the failure mode that motivated this module. Traced on
2026-09-08: a blackboard lecture in which the lecturer reads his own writing
aloud is maximally redundant, but PP-OCR models are trained on printed text, so
recognition of handwriting is largely rejected. Those rejected regions were
being discarded silently, the frame recorded as carrying no text, and the video
scored 0 with status OK -- which this project defines as ideal design.

The direction of the failure is what makes it serious: it does not add noise, it
flatters exactly the videos that most violate the principle.
"""

from __future__ import annotations

from mmpquant.extract.ocr import (
    RELIABILITY_NO_TEXT,
    RELIABILITY_OK,
    RELIABILITY_UNREADABLE,
    OcrFrameResult,
    assess_recognition_reliability,
)
from mmpquant.types import OcrLine


def readable_line(text: str = "some readable slide text") -> OcrLine:
    return OcrLine(text=text, confidence=0.95, bbox=(0, 0, 400, 40))


def frame_read(n: int = 4) -> OcrFrameResult:
    """A frame whose text was detected and successfully read."""
    return OcrFrameResult(lines=tuple(readable_line() for _ in range(n)),
                          n_detected=n, n_rejected=0)


def frame_unreadable(n: int = 8, conf: float = 0.31) -> OcrFrameResult:
    """A frame where text was detected but recognition was rejected."""
    return OcrFrameResult(lines=(), n_detected=n, n_rejected=n,
                          rejected_confidence_mean=conf)


def frame_blank() -> OcrFrameResult:
    """A frame genuinely carrying no text."""
    return OcrFrameResult(lines=(), n_detected=0, n_rejected=0)


class TestFrameResult:
    def test_recovery_rate_when_all_read(self):
        assert frame_read(4).detection_recovery_rate == 1.0

    def test_recovery_rate_when_none_read(self):
        assert frame_unreadable(8).detection_recovery_rate == 0.0

    def test_blank_frame_is_not_a_recovery_failure(self):
        # Nothing was detected, so nothing failed to be read. Reporting 0 here
        # would mark every text-free frame as a recognition problem.
        assert frame_blank().detection_recovery_rate == 1.0

    def test_partial_recovery(self):
        r = OcrFrameResult(lines=(readable_line(),), n_detected=4, n_rejected=3)
        assert r.detection_recovery_rate == 0.25


class TestBlackboardSignature:
    def test_detected_but_unreadable_is_flagged(self):
        # THE case. Text on screen throughout, almost none of it readable.
        summary = assess_recognition_reliability([frame_unreadable() for _ in range(20)])
        assert summary["reliability"] == RELIABILITY_UNREADABLE

    def test_the_flag_says_a_low_score_is_not_evidence_of_good_design(self):
        # The note is what stops a 0 being written up as a well-designed video,
        # so its content is part of the contract.
        summary = assess_recognition_reliability([frame_unreadable() for _ in range(20)])
        note = summary["note"].lower()
        assert "could not be read" in note
        assert "not that the video is free of redundancy" in note

    def test_partial_readability_still_flags(self):
        # Mixed board and slides: half the frames unreadable is still unreliable.
        frames = [frame_unreadable() for _ in range(12)] + [frame_read() for _ in range(8)]
        summary = assess_recognition_reliability(frames)
        assert summary["reliability"] == RELIABILITY_UNREADABLE

    def test_recovery_rate_is_reported_for_inspection(self):
        frames = [frame_unreadable(n=10) for _ in range(10)]
        summary = assess_recognition_reliability(frames)
        assert summary["n_regions_detected"] == 100
        assert summary["n_regions_recovered"] == 0
        assert summary["detection_recovery_rate"] == 0.0


class TestCleanCases:
    def test_readable_slides_are_ok(self):
        summary = assess_recognition_reliability([frame_read() for _ in range(20)])
        assert summary["reliability"] == RELIABILITY_OK

    def test_genuinely_text_free_video_is_distinguished_from_unreadable(self):
        # THE distinction the whole module exists to preserve. Both produce no
        # text, but only one of them means the measurement failed.
        blank = assess_recognition_reliability([frame_blank() for _ in range(20)])
        unreadable = assess_recognition_reliability([frame_unreadable() for _ in range(20)])
        assert blank["reliability"] == RELIABILITY_NO_TEXT
        assert unreadable["reliability"] == RELIABILITY_UNREADABLE
        assert blank["reliability"] != unreadable["reliability"]

    def test_occasional_rejection_does_not_flag(self):
        # Every real video rejects the odd region. That is not a failure.
        frames = [
            OcrFrameResult(lines=(readable_line(),) * 4, n_detected=5, n_rejected=1)
            for _ in range(20)
        ]
        assert assess_recognition_reliability(frames)["reliability"] == RELIABILITY_OK

    def test_sparse_text_is_not_flagged_as_unreadable(self):
        # An animated video with text in only a few frames, all read fine.
        frames = [frame_blank() for _ in range(18)] + [frame_read() for _ in range(2)]
        assert assess_recognition_reliability(frames)["reliability"] == RELIABILITY_OK

    def test_no_frames_at_all(self):
        assert assess_recognition_reliability([])["reliability"] == RELIABILITY_NO_TEXT


class TestThresholds:
    def test_recovery_threshold_is_configurable(self):
        frames = [
            OcrFrameResult(lines=(readable_line(),) * 3, n_detected=10, n_rejected=7)
            for _ in range(20)
        ]  # 30% recovery
        strict = assess_recognition_reliability(frames, min_recovery_rate=0.5)
        lenient = assess_recognition_reliability(frames, min_recovery_rate=0.2)
        assert strict["reliability"] == RELIABILITY_UNREADABLE
        assert lenient["reliability"] == RELIABILITY_OK

    def test_detection_share_threshold_is_configurable(self):
        # Text detected in only 10% of frames, unreadable in those.
        frames = [frame_blank() for _ in range(18)] + [frame_unreadable() for _ in range(2)]
        lenient = assess_recognition_reliability(frames, min_frames_with_detections=0.2)
        strict = assess_recognition_reliability(frames, min_frames_with_detections=0.05)
        assert lenient["reliability"] == RELIABILITY_OK
        assert strict["reliability"] == RELIABILITY_UNREADABLE


class TestDurationWeighting:
    """Defect 9.1: the detection share must be measured in time, not in groups.

    Each result handed to the assessor is one deduplicated group, and groups
    cover wildly unequal spans. A blackboard held still for a minute collapses to
    a single group; a minute of camera movement over the lecturer produces a
    hundred. Counting groups therefore undercounts precisely the static,
    text-bearing content the handwriting safeguard exists to catch, and the
    safeguard stops firing on its own motivating case.
    """

    # 3 board holds of 50s each, then 40 half-second frames of camera movement:
    # the board is on screen 88% of the lecture, and none of it can be read.
    BOARD = [frame_unreadable() for _ in range(3)] + [frame_blank() for _ in range(40)]
    BOARD_WEIGHTS = [50.0] * 3 + [0.5] * 40

    def test_group_counting_misses_the_blackboard(self):
        # Documents the defect: by group count the board is 7% of the video, below
        # the detection-share gate, so the unreadable branch never runs.
        summary = assess_recognition_reliability(self.BOARD)
        assert summary["detection_share"] < 0.2
        assert summary["reliability"] == RELIABILITY_OK  # wrong, and why weights exist

    def test_duration_weighting_catches_it(self):
        summary = assess_recognition_reliability(self.BOARD, weights=self.BOARD_WEIGHTS)
        assert summary["detection_share"] > 0.8
        assert summary["reliability"] == RELIABILITY_UNREADABLE

    def test_both_bases_are_reported_for_audit(self):
        summary = assess_recognition_reliability(self.BOARD, weights=self.BOARD_WEIGHTS)
        assert summary["detection_share_basis"] == "duration"
        assert summary["detection_share_by_group"] < 0.1
        assert summary["detection_share_by_duration"] > 0.8

    def test_basis_is_declared_when_no_weights_given(self):
        # A share computed over groups is not a statement about the video, so the
        # summary has to say which question it answered.
        summary = assess_recognition_reliability(self.BOARD)
        assert summary["detection_share_basis"] == "group_count"
        assert summary["detection_share_by_duration"] is None

    def test_equal_weights_match_group_counting(self):
        frames = [frame_read() for _ in range(6)] + [frame_blank() for _ in range(4)]
        weighted = assess_recognition_reliability(frames, weights=[1.0] * 10)
        counted = assess_recognition_reliability(frames)
        assert weighted["detection_share"] == counted["detection_share"] == 0.6

    def test_short_text_moment_in_a_long_silent_video_is_not_inflated(self):
        # The mirror case. One 2s title card in a 200s text-free animation is 50%
        # of the groups but 1% of the running time; only the latter is meaningful.
        frames = [frame_read(), frame_blank()]
        summary = assess_recognition_reliability(frames, weights=[2.0, 198.0])
        assert summary["detection_share_by_group"] == 0.5
        assert summary["detection_share"] == 0.01

    def test_weights_must_match_results(self):
        import pytest

        with pytest.raises(ValueError, match="weights has 2 entries for 3 results"):
            assess_recognition_reliability([frame_read()] * 3, weights=[1.0, 2.0])

    def test_zero_total_duration_falls_back_to_group_counting(self):
        # Degenerate but reachable: a zero-length tail group. Must not divide by 0.
        summary = assess_recognition_reliability([frame_read()], weights=[0.0])
        assert summary["detection_share_basis"] == "group_count"
        assert summary["reliability"] == RELIABILITY_OK
