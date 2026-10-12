"""Segment aggregation and annotation sheets.

Segments are the unit the supervisor annotates, so these tests guard the
property everything else depends on: the segments a rater is asked to judge and
the segments the system reports must be identical, and the sheet handed to a
rater must not leak the system's answer.
"""

from __future__ import annotations

import csv
import itertools

import pytest

from mmpquant.scoring.redundancy import RedundancyScorer
from mmpquant.segments import (
    aggregate_to_segments,
    aggregate_video,
    segment_bounds,
    write_annotation_sheet,
    write_segment_scores,
)

from .conftest import REDUNDANT_ASR, REDUNDANT_OCR, ConstantEmbedder, make_window


def scored(windows):
    s = RedundancyScorer(embedder=ConstantEmbedder(0.0))
    return [s.score_window(w) for w in windows]


def win(i, ocr, asr):
    return make_window(ocr, asr, index=i, start=i * 10.0, end=(i + 1) * 10.0)


class TestSegmentBounds:
    def test_exact_multiple(self):
        assert segment_bounds(120.0, 30.0) == [
            (0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0)
        ]

    def test_partial_tail_is_kept(self):
        assert segment_bounds(150.0, 60.0)[-1] == (120.0, 150.0)

    def test_typical_supervisor_case(self):
        # 2-3 minute video at the default 30 s: 4-6 segments to annotate.
        assert len(segment_bounds(120.0, 30.0)) == 4
        assert len(segment_bounds(180.0, 30.0)) == 6

    def test_zero_duration(self):
        assert segment_bounds(0.0, 30.0) == []

    def test_segments_tile_without_gaps(self):
        bounds = segment_bounds(155.0, 30.0)
        for (_, prev_end), (next_start, _) in itertools.pairwise(bounds):
            assert prev_end == next_start
        assert bounds[0][0] == 0.0
        assert bounds[-1][1] == 155.0


class TestAggregation:
    def test_window_counts_towards_exactly_one_segment(self):
        # 10 s windows into 30 s segments: three windows each, no double count.
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(6)])
        segs = aggregate_to_segments(ws, 60.0, 30.0)
        assert len(segs) == 2
        assert [s.n_windows for s in segs] == [3, 3]
        assert sum(s.n_windows for s in segs) == 6

    def test_segment_score_is_mean_of_its_windows(self):  # all three assessed
        # verbatim 1.0 -> 60.0 for redundant windows; semantic pinned to 0.
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(3)])
        segs = aggregate_to_segments(ws, 30.0, 30.0)
        assert segs[0].score == pytest.approx(60.0)

    def test_segments_differ_from_each_other(self):
        # First segment redundant, second clean: the whole point of segment
        # level reporting is that these must not be averaged together.
        windows = [win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(3)]
        windows += [win(i, "alpha beta gamma delta epsilon", "one two three")
                    for i in range(3, 6)]
        segs = aggregate_to_segments(scored(windows), 60.0, 30.0)
        assert segs[0].score == pytest.approx(60.0)
        assert segs[1].score == pytest.approx(0.0)

    def test_segment_with_no_qualifying_window_scores_zero(self):
        # Assessed, nothing redundant found: 0, matching the video-level rule.
        ws = scored([win(i, "", "narration present here") for i in range(3)])
        segs = aggregate_to_segments(ws, 30.0, 30.0)
        assert segs[0].score == 0.0
        assert segs[0].status == "OK"
        assert segs[0].n_windows_qualifying == 0

    def test_unassessed_window_counts_zero_inside_a_segment(self):
        # One redundant window, two with nothing on screen: the empty two are
        # real zeros in the average (08 Oct 2026), not dropped from it.
        windows = [win(0, REDUNDANT_OCR, REDUNDANT_ASR)]
        windows += [win(i, "", "narration present here") for i in (1, 2)]
        seg = aggregate_to_segments(scored(windows), 30.0, 30.0)[0]
        assert seg.score == pytest.approx(20.0)
        assert seg.assessed_score == pytest.approx(60.0)
        assert seg.coverage == pytest.approx(1 / 3, abs=1e-4)
        assert seg.score == pytest.approx(seg.assessed_score * seg.coverage, abs=0.01)

    def test_components_share_the_scores_denominator(self):
        # score = 100 × (w_v·verbatim + w_s·semantic) must hold per segment.
        windows = [win(0, REDUNDANT_OCR, REDUNDANT_ASR)]
        windows += [win(i, "", "narration present here") for i in (1, 2)]
        seg = aggregate_to_segments(scored(windows), 30.0, 30.0)[0]
        assert seg.score == pytest.approx(100 * (0.6 * seg.verbatim + 0.4 * seg.semantic), abs=0.01)

    def test_segment_holding_no_windows_is_na(self):
        segs = aggregate_to_segments([], 30.0, 30.0)
        assert segs[0].score is None
        assert segs[0].status == "N/A"

    def test_window_provenance_recorded(self):
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(3)])
        segs = aggregate_to_segments(ws, 30.0, 30.0)
        assert segs[0].window_indices == (0, 1, 2)

    def test_timecode_is_human_readable(self):
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(12)])
        segs = aggregate_to_segments(ws, 120.0, 30.0)
        assert segs[0].timecode == "0:00-0:30"
        assert segs[2].timecode == "1:00-1:30"


class TestAnnotationSheet:
    def test_sheet_has_one_row_per_segment(self, tmp_path):
        path = write_annotation_sheet("demo", 120.0, 30.0, tmp_path / "s.csv")
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 4

    def test_sheet_boundaries_match_the_scorer(self, tmp_path):
        # THE property that makes the join possible without reconciliation.
        path = write_annotation_sheet("demo", 155.0, 30.0, tmp_path / "s.csv")
        rows = list(csv.DictReader(path.open()))
        bounds = segment_bounds(155.0, 30.0)
        assert len(rows) == len(bounds)
        for row, (start, end) in zip(rows, bounds, strict=True):
            assert float(row["start_s"]) == pytest.approx(start)
            assert float(row["end_s"]) == pytest.approx(end)

    def test_sheet_carries_no_system_score(self, tmp_path):
        # Showing a rater the machine's answer makes them anchor to it, and the
        # correlation then measures agreement with the anchor.
        path = write_annotation_sheet("demo", 60.0, 30.0, tmp_path / "s.csv")
        header = path.read_text().splitlines()[0]
        assert "system" not in header.lower()
        assert "score" in header.lower()  # blank columns for the raters

    def test_one_column_per_rater(self, tmp_path):
        path = write_annotation_sheet(
            "demo", 60.0, 30.0, tmp_path / "s.csv", raters=("prof", "colleague")
        )
        header = next(iter(csv.reader(path.open())))
        assert "prof_score" in header
        assert "colleague_score" in header
        assert "prof_notes" in header

    def test_score_columns_start_empty(self, tmp_path):
        path = write_annotation_sheet("demo", 60.0, 30.0, tmp_path / "s.csv")
        for row in csv.DictReader(path.open()):
            assert row["rater_1_score"] == ""
            assert row["rater_2_score"] == ""


class TestSegmentScoreExport:
    def test_export_joins_on_video_and_segment_index(self, tmp_path):
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(6)])
        segs = aggregate_to_segments(ws, 60.0, 30.0)
        path = write_segment_scores(segs, "demo", tmp_path / "sys.csv")
        rows = list(csv.DictReader(path.open()))
        assert [r["video_id"] for r in rows] == ["demo", "demo"]
        assert [r["segment_index"] for r in rows] == ["0", "1"]
        assert all(r["system_score"] for r in rows)

    def test_export_and_sheet_align_row_for_row(self, tmp_path):
        # The two files must be joinable without any reconciliation step.
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(12)])
        segs = aggregate_to_segments(ws, 120.0, 30.0)
        sys_path = write_segment_scores(segs, "demo", tmp_path / "sys.csv")
        sheet = write_annotation_sheet("demo", 120.0, 30.0, tmp_path / "sheet.csv")
        sys_rows = list(csv.DictReader(sys_path.open()))
        sheet_rows = list(csv.DictReader(sheet.open()))
        assert len(sys_rows) == len(sheet_rows)
        for a, b in zip(sys_rows, sheet_rows, strict=True):
            assert (a["video_id"], a["segment_index"]) == (b["video_id"], b["segment_index"])
            assert a["timecode"] == b["timecode"]


def mixed_video(n=18, redundant=(12, 13)):
    """The Test_13 shape: a long video, redundant text in a few windows only."""
    return scored([
        win(i, REDUNDANT_OCR, REDUNDANT_ASR) if i in redundant
        else win(i, "", "narration present here")
        for i in range(n)
    ])


class TestWholeVideoScore:
    """08 Oct 2026: one averaging rule at every level. A window with nothing
    on screen to assess counts 0 in the video score exactly as in its
    segment's, so the video score is the mean of the segment scores and does
    not depend on where the segment grid falls.
    """

    def test_unassessed_windows_count_as_zero(self):
        agg = aggregate_video(mixed_video())
        assert agg.score == pytest.approx(60.0 * 2 / 18, abs=0.01)
        assert agg.assessed_score == pytest.approx(60.0)
        assert agg.coverage == pytest.approx(2 / 18, abs=1e-4)

    def test_score_is_assessed_score_times_coverage(self):
        agg = aggregate_video(mixed_video())
        assert agg.score == pytest.approx(agg.assessed_score * agg.coverage, abs=0.01)

    def test_equals_the_duration_weighted_mean_of_segment_scores(self):
        ws = mixed_video()
        segs = aggregate_to_segments(ws, 180.0, 30.0)
        mean = sum(s.score * s.measured_s for s in segs) / sum(s.measured_s for s in segs)
        assert aggregate_video(ws).score == pytest.approx(mean, abs=0.01)

    @pytest.mark.parametrize("redundant", [(0,), (2, 3), (12, 13), (5, 6, 7, 16)])
    def test_does_not_depend_on_the_segment_grid(self, redundant):
        # The defect this replaces: the old mixed rule moved Test_27 between
        # 6.80 and 12.43 when only the 30 s grid moved. The segment mean must
        # now be the same for any segment length.
        ws = mixed_video(redundant=redundant)
        means = set()
        for length in (20.0, 30.0, 60.0, 90.0):
            segs = aggregate_to_segments(ws, 180.0, length)
            mean = sum(s.score * s.measured_s for s in segs) / sum(s.measured_s for s in segs)
            means.add(round(mean, 1))
        assert len(means) == 1
        assert means.pop() == pytest.approx(aggregate_video(ws).score, abs=0.05)

    def test_a_shorter_window_is_weighted_by_its_own_duration(self):
        ws = scored([
            make_window("", "narration present here", index=0, start=0.0, end=10.0),
            make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=1, start=10.0, end=15.0),
        ])
        assert aggregate_video(ws).score == pytest.approx(60.0 * 5 / 15, abs=0.01)

    def test_fully_covered_video_equals_the_assessed_average(self):
        ws = scored([win(i, REDUNDANT_OCR, REDUNDANT_ASR) for i in range(6)])
        agg = aggregate_video(ws)
        assert agg.score == agg.assessed_score == pytest.approx(60.0)
        assert agg.coverage == 1.0

    def test_nothing_on_screen_anywhere_scores_zero_not_none(self):
        # User decision 08 Oct 2026: a text-free video was assessed and holds
        # no redundancy -> 0 (CLAUDE.md §2.7 said N/A; superseded).
        ws = scored([win(i, "", "narration present here") for i in range(6)])
        agg = aggregate_video(ws)
        assert agg.score == 0.0
        assert agg.assessed_score is None
        assert agg.coverage == 0.0

    def test_no_windows_at_all_returns_none(self):
        agg = aggregate_video([])
        assert agg.score is None and agg.assessed_score is None
