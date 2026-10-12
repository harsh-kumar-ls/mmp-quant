"""ASR-OCR time alignment tests.

Required by the spec. These pin down the ±3 s tolerance semantics, which are
otherwise ambiguous and are the single easiest place for a silent scoring bug to
live.
"""

from __future__ import annotations

import itertools

import pytest

from mmpquant.extract.align import (
    WindowingParams,
    blocks_from_groups,
    build_windows,
    window_bounds,
)
from mmpquant.extract.ocr.dedup import FrameGroup
from mmpquant.types import AsrSegment, AsrWord, OcrBlock, OcrLine, Transcript


def words(*specs: tuple[float, float, str]) -> tuple[AsrWord, ...]:
    return tuple(AsrWord(start=s, end=e, text=t) for s, e, t in specs)


def transcript_of(*specs: tuple[float, float, str], duration: float = 30.0) -> Transcript:
    ws = words(*specs)
    seg = AsrSegment(
        start=ws[0].start if ws else 0.0,
        end=ws[-1].end if ws else 0.0,
        text=" ".join(w.text for w in ws),
        words=ws,
    )
    return Transcript(
        segments=(seg,) if ws else (),
        language="en",
        language_probability=0.99,
        duration=duration,
    )


def block(start: float, end: float, *lines: str) -> OcrBlock:
    return OcrBlock(
        start=start,
        end=end,
        lines=tuple(OcrLine(text=t, confidence=0.95) for t in lines),
        source_frame_time=(start + end) / 2,
        n_frames_collapsed=1,
    )


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


class TestWindowBounds:
    def test_exact_multiple(self):
        assert window_bounds(30.0, WindowingParams(size_s=10.0)) == [
            (0.0, 10.0), (10.0, 20.0), (20.0, 30.0)
        ]

    def test_partial_final_window_kept_by_default(self):
        bounds = window_bounds(25.0, WindowingParams(size_s=10.0))
        assert bounds[-1] == (20.0, 25.0)

    def test_partial_final_window_can_be_dropped(self):
        params = WindowingParams(size_s=10.0, keep_partial_final_window=False)
        assert window_bounds(25.0, params) == [(0.0, 10.0), (10.0, 20.0)]

    def test_windows_are_contiguous_and_non_overlapping(self):
        bounds = window_bounds(47.0, WindowingParams(size_s=10.0))
        for (_, prev_end), (next_start, _) in itertools.pairwise(bounds):
            assert prev_end == next_start

    def test_video_shorter_than_one_window(self):
        assert window_bounds(4.0, WindowingParams(size_s=10.0)) == [(0.0, 4.0)]

    def test_zero_duration_yields_no_windows(self):
        assert window_bounds(0.0, WindowingParams()) == []

    def test_configurable_window_size(self):
        assert len(window_bounds(30.0, WindowingParams(size_s=5.0))) == 6


class TestWindowingParamsValidation:
    def test_rejects_non_positive_size(self):
        with pytest.raises(ValueError):
            WindowingParams(size_s=0.0)

    def test_rejects_negative_tolerance(self):
        with pytest.raises(ValueError):
            WindowingParams(tolerance_s=-1.0)

    def test_rejects_unknown_expand_mode(self):
        with pytest.raises(ValueError):
            WindowingParams(expand="sideways")

    def test_tolerance_routed_to_the_configured_side(self):
        asr_only = WindowingParams(tolerance_s=3.0, expand="asr")
        assert (asr_only.asr_tolerance, asr_only.ocr_tolerance) == (3.0, 0.0)

        ocr_only = WindowingParams(tolerance_s=3.0, expand="ocr")
        assert (ocr_only.asr_tolerance, ocr_only.ocr_tolerance) == (0.0, 3.0)

        both = WindowingParams(tolerance_s=3.0, expand="both")
        assert (both.asr_tolerance, both.ocr_tolerance) == (3.0, 3.0)


# --------------------------------------------------------------------------- #
# ASR assignment
# --------------------------------------------------------------------------- #


class TestAsrAssignment:
    def test_word_lands_in_its_own_window(self):
        tr = transcript_of((2.0, 2.5, "hello"))
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=0.0))
        assert w[0].asr_text == "hello"
        assert w[1].asr_text == ""

    def test_assignment_uses_word_midpoint(self):
        # Word spans 9.0-11.0, midpoint 10.0. With zero tolerance it belongs to
        # window [10,20) only -- it must not be duplicated into both windows.
        tr = transcript_of((9.0, 11.0, "straddle"))
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=0.0))
        assert w[0].asr_text == ""
        assert w[1].asr_text == "straddle"

    def test_tolerance_pulls_in_narration_just_outside_the_window(self):
        # Narration at 11.5 s is outside [0,10) but inside the +3 s pool.
        tr = transcript_of((11.0, 12.0, "lagging"))
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=3.0, expand="asr"))
        assert "lagging" in w[0].asr_text
        assert "lagging" in w[1].asr_text  # legitimately shared: that IS tolerance

    def test_tolerance_of_zero_partitions_narration_exactly_once(self):
        tr = transcript_of(
            (1.0, 1.5, "a"), (11.0, 11.5, "b"), (21.0, 21.5, "c")
        )
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=0.0))
        assert [win.asr_text for win in w] == ["a", "b", "c"]

    def test_expand_ocr_leaves_narration_hard_bounded(self):
        tr = transcript_of((11.0, 12.0, "lagging"))
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=3.0, expand="ocr"))
        assert w[0].asr_text == ""

    def test_pool_bounds_are_recorded_for_the_audit_trail(self):
        tr = transcript_of((1.0, 1.5, "x"))
        w = build_windows(tr, (), 30.0, WindowingParams(tolerance_s=3.0, expand="asr"))
        assert (w[1].asr_pool_start, w[1].asr_pool_end) == (7.0, 23.0)

    def test_word_order_is_preserved(self):
        tr = transcript_of((1.0, 1.2, "energy"), (1.3, 1.5, "is"), (1.6, 1.9, "released"))
        w = build_windows(tr, (), 30.0, WindowingParams())
        assert w[0].asr_text == "energy is released"

    def test_silent_video_yields_windows_with_empty_narration(self):
        tr = Transcript(segments=(), language=None, language_probability=None, duration=30.0)
        w = build_windows(tr, (), 30.0, WindowingParams())
        assert len(w) == 3
        assert all(win.asr_text == "" for win in w)


# --------------------------------------------------------------------------- #
# OCR assignment
# --------------------------------------------------------------------------- #


class TestOcrAssignment:
    def test_block_assigned_to_its_window(self):
        w = build_windows(transcript_of(), (block(0.0, 5.0, "Slide one"),), 30.0,
                          WindowingParams(tolerance_s=0.0))
        assert w[0].ocr_text == "Slide one"
        assert w[1].ocr_text == ""

    def test_long_slide_counts_in_every_window_it_covers(self):
        # A slide held for the whole video is on screen during all 3 windows.
        # Midpoint assignment would wrongly place it in only one -- hence
        # intersection semantics for OCR blocks.
        w = build_windows(transcript_of(), (block(0.0, 30.0, "Held slide"),), 30.0,
                          WindowingParams(tolerance_s=0.0))
        assert [win.ocr_text for win in w] == ["Held slide"] * 3

    def test_duplicate_lines_across_blocks_are_kept_once(self):
        # Two blocks = same slide re-detected after a cursor moved. Emitting the
        # shared line twice would double the on-screen word count and could push
        # a short signaling label past min_onscreen_words.
        blocks = (
            block(0.0, 4.0, "Key Terms", "ATP synthase"),
            block(4.0, 9.0, "Key Terms", "ATP synthase"),
        )
        w = build_windows(transcript_of(), blocks, 30.0, WindowingParams(tolerance_s=0.0))
        assert w[0].ocr_text == "Key Terms\nATP synthase"

    def test_distinct_lines_across_blocks_are_concatenated_in_order(self):
        blocks = (block(0.0, 4.0, "First"), block(4.0, 9.0, "Second"))
        w = build_windows(transcript_of(), blocks, 30.0, WindowingParams(tolerance_s=0.0))
        assert w[0].ocr_text == "First\nSecond"

    def test_block_provenance_is_recorded(self):
        blocks = (block(0.0, 4.0, "A"), block(15.0, 20.0, "B"))
        w = build_windows(transcript_of(), blocks, 30.0, WindowingParams(tolerance_s=0.0))
        assert w[0].ocr_block_indices == (0,)
        assert w[1].ocr_block_indices == (1,)

    def test_empty_block_contributes_no_text(self):
        w = build_windows(transcript_of(), (block(0.0, 10.0),), 30.0, WindowingParams())
        assert w[0].ocr_text == ""

    def test_text_free_video_yields_windows_with_empty_ocr(self):
        tr = transcript_of((1.0, 2.0, "talking"))
        w = build_windows(tr, (), 30.0, WindowingParams())
        assert all(win.ocr_text == "" for win in w)


# --------------------------------------------------------------------------- #
# Frame groups -> time-extended blocks
# --------------------------------------------------------------------------- #


class TestBlocksFromGroups:
    def test_blocks_tile_the_timeline_without_gaps(self):
        # A gap would mean a window silently misses on-screen text that was
        # actually present -- an under-report of redundancy.
        frame_times = [i * 0.5 for i in range(20)]  # 2 fps, 10 s
        groups = [FrameGroup(0, 7, 3), FrameGroup(8, 19, 13)]
        blocks = blocks_from_groups(frame_times, groups, [(), ()], video_duration=10.0)
        assert blocks[0].start == 0.0
        assert blocks[0].end == blocks[1].start == 4.0
        assert blocks[1].end == 10.0

    def test_last_block_extends_to_video_end(self):
        # Sampling stops before the true end (last frame at 9.5 s of a 10 s
        # video); the final slide is still on screen until 10.0 s.
        frame_times = [i * 0.5 for i in range(20)]
        blocks = blocks_from_groups(frame_times, [FrameGroup(0, 19, 10)], [()], 10.0)
        assert blocks[0].end == 10.0

    def test_representative_frame_time_is_carried_through(self):
        frame_times = [i * 0.5 for i in range(20)]
        blocks = blocks_from_groups(frame_times, [FrameGroup(0, 19, 10)], [()], 10.0)
        assert blocks[0].source_frame_time == 5.0

    def test_frames_collapsed_count_is_carried_through(self):
        frame_times = [i * 0.5 for i in range(20)]
        blocks = blocks_from_groups(frame_times, [FrameGroup(0, 19, 10)], [()], 10.0)
        assert blocks[0].n_frames_collapsed == 20

    def test_lines_are_attached_to_the_right_block(self):
        frame_times = [0.0, 0.5, 1.0, 1.5]
        groups = [FrameGroup(0, 1, 0), FrameGroup(2, 3, 2)]
        lines = [(OcrLine("first", 0.9),), (OcrLine("second", 0.9),)]
        blocks = blocks_from_groups(frame_times, groups, lines, 2.0)
        assert blocks[0].text == "first"
        assert blocks[1].text == "second"

    def test_no_groups_yields_no_blocks(self):
        assert blocks_from_groups([], [], [], 10.0) == ()


class TestDegenerateTailWindow:
    """Regression: a video of 10.0001 s produced a second window of 0.0001 s.

    Such a window can hold neither narration nor a slide, but it enters the
    duration-weighted mean, and divides by ~0 if it is the only qualifying one.
    """

    def test_sub_tenth_second_tail_is_discarded(self):
        bounds = window_bounds(10.0001, WindowingParams(size_s=10.0))
        assert bounds == [(0.0, 10.0)]

    def test_meaningful_tail_is_still_kept(self):
        bounds = window_bounds(12.5, WindowingParams(size_s=10.0))
        assert bounds[-1] == (10.0, 12.5)

    def test_every_window_has_usable_duration(self):
        for duration in (10.0001, 20.00001, 33.3, 47.0, 100.05):
            for start, end in window_bounds(duration, WindowingParams(size_s=10.0)):
                assert end - start >= 0.1
