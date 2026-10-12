"""OCR deduplication tests.

Required by the spec: "static slides must not be re-OCR'd on every frame".
The grouping logic is pure, so these tests use synthetic signature arrays and
need no video, no images and no OCR engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmpquant.extract.ocr.dedup import (
    METHOD_CHANGED_PIXELS,
    METHOD_MEAN_ABS_DIFF,
    FrameGroup,
    dedup_ratio,
    frame_distance,
    group_frames,
)

SHAPE = (100, 100)  # 10 000 pixels, so a pixel count reads directly as a ratio


def flat(value: float, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def slide(dark_from: int, dark_count: int = 125, ink: float = 0.6) -> np.ndarray:
    """A white 'slide' with `dark_count` ink pixels starting at `dark_from`.

    Two slides built with disjoint ink ranges reproduce the measured real-world
    numbers: ~2.5 % of pixels change, but the mean absolute difference is only
    ~0.010 because the other 97.5 % of the frame is unchanged white.
    """
    sig = flat(1.0).ravel().copy()
    sig[dark_from : dark_from + dark_count] = ink
    return sig.reshape(SHAPE)


class TestFrameDistance:
    def test_identical_is_zero_under_both_methods(self):
        a = flat(0.5)
        assert frame_distance(a, a, METHOD_CHANGED_PIXELS) == 0.0
        assert frame_distance(a, a, METHOD_MEAN_ABS_DIFF) == 0.0

    def test_opposite_extremes_is_one_under_both_methods(self):
        a, b = flat(0.0), flat(1.0)
        assert frame_distance(a, b, METHOD_CHANGED_PIXELS) == pytest.approx(1.0)
        assert frame_distance(a, b, METHOD_MEAN_ABS_DIFF) == pytest.approx(1.0)

    def test_changed_pixel_ratio_counts_pixels_not_magnitude(self):
        # 2 500 of 10 000 pixels move by 0.4 -> ratio 0.25 regardless of the 0.4.
        a = flat(1.0)
        b = a.copy().ravel()
        b[:2500] = 0.6
        b = b.reshape(SHAPE)
        assert frame_distance(a, b, METHOD_CHANGED_PIXELS) == pytest.approx(0.25)

    def test_pixel_tolerance_absorbs_sub_threshold_jitter(self):
        # Every pixel moves, but only by 0.05 -- below the 0.1 tolerance, so this
        # is compression noise, not a layout change.
        assert frame_distance(flat(0.50), flat(0.55), METHOD_CHANGED_PIXELS,
                              pixel_tolerance=0.1) == 0.0

    def test_mean_abs_diff_still_available(self):
        assert frame_distance(flat(0.2), flat(0.5), METHOD_MEAN_ABS_DIFF) == pytest.approx(0.3)

    def test_shape_mismatch_treated_as_maximally_different(self):
        # Resolution change mid-video: cannot compare, must not merge.
        assert frame_distance(flat(0.5, (8, 8)), flat(0.5, (4, 4))) == 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown dedup method"):
            frame_distance(flat(0.1), flat(0.2), "perceptual_hash")


class TestSparseTextSlideChange:
    """Regression tests for the bug found on the first end-to-end run.

    Mean absolute difference merged three visibly different slides into one
    block, so two of the three were never OCR'd and the redundant fixture scored
    54 instead of ~100. These pin the measurement that diagnosed it.
    """

    def test_mean_abs_diff_is_diluted_below_a_usable_threshold(self):
        # THE bug: a real slide change registers only ~0.010, which is under the
        # 0.02 default the metric shipped with -- and close enough to
        # compression noise that no safe threshold exists.
        d = frame_distance(slide(0), slide(125), METHOD_MEAN_ABS_DIFF)
        assert d == pytest.approx(0.010, abs=0.002)
        assert d < 0.02, "this is the dilution that caused the missed slide change"

    def test_changed_pixel_ratio_detects_the_same_change_clearly(self):
        d = frame_distance(slide(0), slide(125), METHOD_CHANGED_PIXELS)
        assert d == pytest.approx(0.025, abs=0.005)
        assert d > 0.005 * 4, "must clear the default threshold with real margin"

    def test_default_grouping_splits_a_sparse_text_slide_change(self):
        sigs = [slide(0)] * 5 + [slide(125)] * 5
        assert len(group_frames(sigs)) == 2

    def test_mean_abs_diff_reproduces_the_original_failure(self):
        # Documents the rejected configuration rather than trusting a comment.
        sigs = [slide(0)] * 5 + [slide(125)] * 5
        merged = group_frames(sigs, threshold=0.02, method=METHOD_MEAN_ABS_DIFF)
        assert len(merged) == 1


class TestGroupFrames:
    def test_empty_input(self):
        assert group_frames([]) == []

    def test_single_frame_is_one_group(self):
        assert group_frames([flat(0.5)]) == [FrameGroup(0, 0, 0)]

    def test_static_slide_collapses_to_one_group(self):
        # THE core requirement: 80 identical frames of one slide -> 1 OCR call.
        groups = group_frames([flat(0.5) for _ in range(80)])
        assert len(groups) == 1
        assert groups[0].n_frames == 80

    def test_compression_noise_does_not_split_a_static_slide(self):
        rng = np.random.default_rng(0)
        base = rng.random(SHAPE).astype(np.float32)
        noisy = [
            np.clip(base + rng.normal(0, 0.01, SHAPE), 0, 1).astype(np.float32)
            for _ in range(30)
        ]
        assert len(group_frames(noisy)) == 1

    def test_slide_change_starts_a_new_group(self):
        sigs = [flat(0.1)] * 4 + [flat(0.9)] * 6
        groups = group_frames(sigs)
        assert len(groups) == 2
        assert (groups[0].first_index, groups[0].last_index) == (0, 3)
        assert (groups[1].first_index, groups[1].last_index) == (4, 9)

    def test_every_frame_distinct_yields_no_deduplication(self):
        # Animated content / talking-head inset: dedup legitimately does nothing.
        # Steps of 0.15 are clearly above the 0.1 pixel tolerance; 0.1 steps
        # would sit exactly on the boundary and give float-dependent results.
        sigs = [flat(i * 0.15) for i in range(7)]
        assert len(group_frames(sigs)) == 7

    def test_slow_reveal_is_split_not_silently_merged(self):
        # THE regression test for the anchor-comparison decision.
        # Each frame darkens 10 more pixels (0.1 % of the frame) than the last,
        # like a bullet list being revealed. Every *consecutive* step is below
        # the threshold, so predecessor-comparison would merge all 5 frames into
        # one group and OCR a single frame for two different slide states.
        # Anchor-comparison bounds the drift and splits correctly.
        sigs = [slide(0, dark_count=i * 10, ink=0.0) for i in range(5)]
        groups = group_frames(sigs, threshold=0.0025)
        assert len(groups) > 1, "cumulative drift must break a group"
        assert (groups[0].first_index, groups[0].last_index) == (0, 2)
        assert (groups[1].first_index, groups[1].last_index) == (3, 4)

    def test_groups_tile_the_input_with_no_gaps_or_overlap(self):
        sigs = [flat(0.1)] * 3 + [flat(0.5)] * 2 + [flat(0.9)] * 4
        covered: list[int] = []
        for g in group_frames(sigs):
            covered.extend(range(g.first_index, g.last_index + 1))
        assert covered == list(range(len(sigs)))

    def test_representative_is_the_middle_frame(self):
        # First and last frames of a group are the ones most likely caught
        # mid-transition; OCR'ing a half-faded frame drops lines.
        assert group_frames([flat(0.1)] * 5)[0].representative_index == 2

    def test_representative_within_group_bounds(self):
        sigs = [flat(0.1)] * 3 + [flat(0.9)] * 4
        for g in group_frames(sigs):
            assert g.first_index <= g.representative_index <= g.last_index

    def test_zero_thresholds_split_on_any_difference(self):
        sigs = [flat(0.10), flat(0.10), flat(0.100001)]
        groups = group_frames(sigs, threshold=0.0, pixel_tolerance=0.0)
        assert len(groups) == 2

    def test_huge_threshold_merges_everything(self):
        assert len(group_frames([flat(0.0), flat(1.0), flat(0.0)], threshold=1.0)) == 1


class TestDedupRatio:
    def test_full_collapse(self):
        assert dedup_ratio(100, 1) == pytest.approx(0.99)

    def test_no_collapse(self):
        assert dedup_ratio(10, 10) == 0.0

    def test_zero_frames_is_zero_not_a_division_error(self):
        assert dedup_ratio(0, 0) == 0.0
