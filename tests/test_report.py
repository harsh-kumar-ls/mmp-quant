"""The Markdown report must never assert something the run does not support.

Build order step 3. The report is the artifact a professor reads, so its claims
carry more weight than the JSON's. Two properties are load-bearing and tested
here rather than eyeballed:

  * N/A is never rendered as a number, and never as zero;
  * a caveat that is true of one run is not printed over a run where it is false
    (the detection-share basis is the live example).
"""

from __future__ import annotations

import pytest

from mmpquant.report import render_markdown


def payload(**over):
    base = {
        "video": "data/videos/demo.mp4",
        "video_id": "demo",
        "duration": 60.0,
        "segment_length_s": 30.0,
        "ocr_recognition_reliability": {
            "reliability": "ok",
            "detection_share": 0.9,
            "detection_share_basis": "duration",
            "n_regions_detected": 10,
            "n_regions_recovered": 10,
            "note": "100% of detected text regions were read successfully",
        },
        "segments": [
            {
                "index": 0, "start": 0.0, "end": 30.0, "timecode": "0:00-0:30",
                "score": 40.0, "status": "OK", "na_reason": None,
                "n_windows": 3, "n_windows_qualifying": 1, "window_indices": [0],
                "verbatim": 0.5, "semantic": 0.25,
            },
            {
                "index": 1, "start": 30.0, "end": 60.0, "timecode": "0:30-1:00",
                "score": 0.0, "status": "OK", "na_reason": None,
                "n_windows": 3, "n_windows_qualifying": 0, "window_indices": [1],
                "verbatim": None, "semantic": None,
            },
        ],
        "result": {
            "principle": "redundancy", "score": 20.0, "status": "OK",
            "na_reason": None, "n_windows_total": 2, "n_windows_qualifying": 1,
            "warnings": [],
            "params": {
                "weights": {"verbatim": 0.6, "semantic": 0.4},
                "min_onscreen_words": 5, "min_narration_words": 1,
                "embedding_model": "all-MiniLM-L6-v2",
                "text_normalisation": {"lowercase": True, "strip_punctuation": False},
            },
            "windows": [
                {
                    "index": 0, "start": 0.0, "end": 10.0, "duration": 10.0,
                    "asr_text": "the mitochondrion is the powerhouse of the cell",
                    "ocr_text": "The mitochondrion is the powerhouse of the cell\nCHAPTER 4",
                    "ocr_sentence_text": "The mitochondrion is the powerhouse of the cell",
                    "ocr_elements": [], "score": 40.0, "qualifies": True,
                    "components": {"verbatim": 0.5, "semantic": 0.25,
                                   "n_label_elements": 1.0},
                    "exclusion_reason": None,
                },
                {
                    "index": 1, "start": 30.0, "end": 40.0, "duration": 10.0,
                    "asr_text": "and so we move on", "ocr_text": "CHAPTER 5",
                    "ocr_sentence_text": "", "ocr_elements": [], "score": None,
                    "qualifies": False, "components": {},
                    "exclusion_reason": "no_onscreen_text",
                },
            ],
        },
        "config": {
            "windowing": {"size_s": 10.0}, "alignment": {"tolerance_s": 3.0},
            "media": {"frame_fps": 2.0}, "asr": {"backend": "faster-whisper",
            "model": "small"}, "ocr": {"backend": "rapidocr"},
        },
        "config_path": "config.yaml",
    }
    base.update(over)
    return base


class TestEvidence:
    def test_worst_segment_quotes_both_texts_side_by_side(self):
        md = render_markdown(payload())
        assert "| On-screen text (scored) | Narration |" in md
        assert "the powerhouse of the cell" in md

    def test_scored_text_is_quoted_not_summarised(self):
        # A reader must be able to disagree with the number from the report alone.
        md = render_markdown(payload())
        assert "The mitochondrion is the powerhouse of the cell" in md

    def test_excluded_labels_are_disclosed_not_silently_dropped(self):
        md = render_markdown(payload())
        assert "classified as labels or key terms" in md

    def test_exclusions_are_counted_with_their_reason(self):
        md = render_markdown(payload())
        assert "no on-screen text of any kind" in md

    def test_segments_are_detailed_worst_first(self):
        p = payload()
        p["segments"][1]["score"] = 90.0
        p["segments"][1]["n_windows_qualifying"] = 1
        md = render_markdown(p, top_n=2)
        assert md.index("Segment 1 —") < md.index("Segment 0 —")


class TestNotAssertingWhatIsNotTrue:
    def test_duration_basis_claims_a_share_of_running_time(self):
        md = render_markdown(payload())
        assert "genuine share of the running time" in md
        assert "not a share of the running time" not in md

    def test_group_basis_is_flagged_as_not_a_share_of_the_video(self):
        p = payload()
        p["ocr_recognition_reliability"]["detection_share_basis"] = "group_count"
        md = render_markdown(p)
        assert "not a share of the running time" in md
        assert "genuine share of the running time" not in md

    def test_missing_basis_does_not_claim_duration_weighting(self):
        # Runs produced before the fix carry no basis field at all.
        p = payload()
        del p["ocr_recognition_reliability"]["detection_share_basis"]
        md = render_markdown(p)
        assert "basis not recorded" in md
        assert "genuine share of the running time" not in md


class TestNAIsNeverZero:
    def test_na_video_renders_na_not_a_number(self):
        p = payload()
        p["result"]["score"] = None
        p["result"]["status"] = "N/A"
        p["result"]["na_reason"] = "no narration detected in any of 6 windows"
        md = render_markdown(p)
        assert "**Score: N/A / 100**" in md
        assert "N/A is not zero" in md
        assert "no narration detected" in md

    def test_zero_scoring_segment_is_not_rendered_as_na(self):
        # Segment 1 scores 0.0: assessed, nothing found. Distinct from N/A.
        md = render_markdown(payload())
        row = next(ln for ln in md.splitlines() if ln.startswith("| 1 | 0:30-1:00"))
        assert "0.00" in row
        assert "N/A" not in row


class TestReliabilityBanner:
    def test_unreadable_ocr_is_banner_not_footnote(self):
        p = payload()
        p["ocr_recognition_reliability"].update(
            reliability="text_present_but_unreadable",
            detection_recovery_rate=0.1,
            note="text was detected across 88% of the video but only 10% could be read",
        )
        md = render_markdown(p)
        head = md[: md.index("## Segment timeline")]
        assert "OCR UNRELIABLE" in head
        assert "not** that the video is free of redundancy" in head

    def test_ok_reliability_prints_no_banner(self):
        assert "OCR UNRELIABLE" not in render_markdown(payload())


class TestMethodology:
    def test_weights_used_for_this_run_are_printed(self):
        p = payload()
        p["result"]["params"]["weights"] = {"verbatim": 0.75, "semantic": 0.25}
        md = render_markdown(p)
        assert "0.75 × ROUGE-L F1" in md
        assert "0.25 × cosine" in md

    def test_thresholds_are_printed(self):
        md = render_markdown(payload())
        for expected in ("| Window size | 10.0 s |", "| Segment length | 30.0 s |",
                         "| Min on-screen sentence words | 5 |",
                         "| Alignment tolerance | ±3.0 s |"):
            assert expected in md

    def test_only_enabled_normalisation_steps_are_listed(self):
        md = render_markdown(payload())
        assert "lowercase" in md
        assert "strip_punctuation" not in md


class TestTableSafety:
    def test_pipes_in_text_cannot_break_the_table(self):
        p = payload()
        p["result"]["windows"][0]["ocr_sentence_text"] = "a | b | c is a table"
        md = render_markdown(p)
        assert r"a \| b \| c is a table" in md

    def test_newlines_are_flattened(self):
        p = payload()
        p["result"]["windows"][0]["ocr_sentence_text"] = "line one\nline two\nline three"
        md = render_markdown(p)
        assert "line one line two line three" in md

    def test_empty_text_renders_a_placeholder_not_a_broken_row(self):
        p = payload()
        p["result"]["windows"][0]["asr_text"] = ""
        md = render_markdown(p)
        assert "_(none)_" in md


class TestNoQualifyingWindows:
    def test_video_with_nothing_scored_still_renders(self):
        p = payload()
        for s in p["segments"]:
            s["score"] = None
            s["n_windows_qualifying"] = 0
        p["result"]["windows"][0]["qualifies"] = False
        p["result"]["windows"][0]["score"] = None
        md = render_markdown(p)
        assert "No segment carried measurable redundancy." in md


@pytest.mark.parametrize("score,expected_filled", [(0.0, 0), (100.0, 24), (50.0, 12)])
def test_bar_is_fixed_scale_so_videos_compare(score, expected_filled):
    from mmpquant.report import _bar

    assert _bar(score).count("█") == expected_filled


def test_floor_scoring_video_does_not_list_empty_worst_offenders():
    # Every segment assessed, nothing found. Three empty subsections headed
    # "worst offender" bury the finding instead of stating it.
    p = payload()
    for s in p["segments"]:
        s["score"] = 0.0
        s["n_windows_qualifying"] = 0
    md = render_markdown(p)
    assert "No segment carried measurable redundancy." in md
    assert "### Segment 0 —" not in md


class TestAggregationNote:
    """08 Oct 2026: when part of the video had nothing to assess, the report
    shows the assessed-only average, the coverage, and how they multiply to
    the reported score; silent when the two figures coincide."""

    def test_absent_fields_print_nothing(self):
        md = render_markdown(payload())
        assert "could be assessed in" not in md

    def test_matching_scores_print_nothing(self):
        p = payload()
        p["assessed_score"], p["coverage"] = p["result"]["score"], 1.0
        assert "could be assessed in" not in render_markdown(p)

    def test_low_coverage_shows_the_arithmetic(self):
        p = payload()
        p["assessed_score"], p["coverage"] = 5.67, 0.2778
        p["result"]["score"] = 1.58
        md = render_markdown(p)
        assert "could be assessed in **28%**" in md
        assert "5.67 × 0.2778 = 1.58" in md

    def test_methodology_states_the_single_averaging_rule(self):
        md = render_markdown(payload())
        assert "a window with nothing to assess counts as 0" in md
        assert "duration-weighted mean of the segment scores" in md

