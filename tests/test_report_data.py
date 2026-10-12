"""`build_report_data` must agree with `render_markdown` — same payload, same
judgement calls, just as primitives instead of text.

Reuses `test_report.py`'s `payload()` fixture rather than a second copy, so the
two report tests are exercising literally the same input.
"""

from __future__ import annotations

from mmpquant.report_data import build_report_data
from tests.test_report import payload


class TestAgreesWithMarkdownReport:
    def test_na_score_stays_none_not_zero(self):
        p = payload()
        p["result"]["score"] = None
        p["result"]["status"] = "N/A"
        p["result"]["na_reason"] = "no narration detected in any of 6 windows"
        data = build_report_data(p)
        assert data["score"] is None
        assert data["status"] == "N/A"
        assert "no narration detected" in data["na_reason"]

    def test_floor_scoring_video_has_no_worst_segments(self):
        p = payload()
        for s in p["segments"]:
            s["score"] = 0.0
            s["n_windows_qualifying"] = 0
        data = build_report_data(p)
        assert data["worst_segments"] == []

    def test_worst_segments_are_sorted_highest_first(self):
        p = payload()
        p["segments"][1]["score"] = 90.0
        p["segments"][1]["n_windows_qualifying"] = 1
        data = build_report_data(p, top_n=2)
        assert [s["segment_index"] for s in data["worst_segments"]] == [1, 0]

    def test_exclusion_reasons_use_the_same_plain_english_labels(self):
        data = build_report_data(payload())
        assert data["exclusions"] == [
            {"reason": "no_onscreen_text", "label": "no on-screen text of any kind", "count": 1}
        ]

    def test_duration_basis_carries_the_same_caveat_phrase_as_the_markdown_report(self):
        data = build_report_data(payload())
        assert data["reliability"]["basis_phrase"] == "weighted by duration"

    def test_missing_basis_is_reported_as_not_recorded_not_silently_assumed(self):
        p = payload()
        del p["ocr_recognition_reliability"]["detection_share_basis"]
        data = build_report_data(p)
        assert data["reliability"]["basis_phrase"] == "basis not recorded"


class TestEvidenceIsQuotedInFull:
    def test_window_evidence_carries_both_texts_untruncated(self):
        data = build_report_data(payload())
        window = data["worst_segments"][0]["windows"][0]
        assert window["onscreen_text"] == (
            "The mitochondrion is the powerhouse of the cell"
        )
        assert "powerhouse of the cell" in window["narration_text"]

    def test_excluded_label_count_travels_with_the_window(self):
        data = build_report_data(payload())
        window = data["worst_segments"][0]["windows"][0]
        assert window["n_labels_excluded"] == 1


class TestMethodologyReflectsTheActualRun:
    def test_weights_are_the_run_s_own_not_defaults(self):
        p = payload()
        p["result"]["params"]["weights"] = {"verbatim": 0.75, "semantic": 0.25}
        data = build_report_data(p)
        assert data["methodology"]["weight_verbatim"] == 0.75
        assert data["methodology"]["weight_semantic"] == 0.25

    def test_thresholds_are_present(self):
        data = build_report_data(payload())
        m = data["methodology"]
        assert m["window_size_s"] == 10.0
        assert m["segment_length_s"] == 30.0
        assert m["min_onscreen_words"] == 5


class TestAssessedScoreAndCoverageAreCarried:
    """08 Oct 2026: the assessed-only average and coverage travel with the
    reported score in the JSON output, as they do in the Markdown report."""

    def test_present_when_the_payload_carries_them(self):
        p = payload()
        p["assessed_score"], p["coverage"] = 5.67, 0.2778
        data = build_report_data(p)
        assert (data["assessed_score"], data["coverage"]) == (5.67, 0.2778)

    def test_absent_payload_fields_are_none_not_missing_keys(self):
        data = build_report_data(payload())
        assert data["assessed_score"] is None and data["coverage"] is None


class TestSegmentBreakdownForTheWebPopup:
    """05 Oct 2026: every segment carries a per-window breakdown so the web UI
    can show the full calculation when a segment is clicked."""

    def test_each_segment_lists_its_windows(self):
        data = build_report_data(payload())
        seg0 = data["segments"][0]
        assert [w["index"] for w in seg0["windows"]] == [0]

    def test_breakdown_carries_text_and_numbers(self):
        p = payload()
        w = p["result"]["windows"][0]
        w["components"].update(verbatim_raw=0.6, verbatim_chance=0.2, semantic_raw=0.8,
                               semantic_chance=0.6)
        w["evidence"] = {"gate": "passed", "null_windows": 5, "sentence_checks": [
            {"sentence": "S", "is_question": False, "content_words": ["a", "b"],
             "shared_words": ["a", "b"], "coverage": 1.0, "chance": 0.0, "passed": True}]}
        b = build_report_data(p)["segments"][0]["windows"][0]
        assert b["narration"].startswith("the mitochondrion")
        assert b["verbatim_raw"] == 0.6 and b["semantic_chance"] == 0.6
        assert b["sentence_checks"][0]["passed"] is True
        assert b["gate"] == "passed"

    def test_excluded_window_says_why(self):
        b = build_report_data(payload())["segments"][1]["windows"][0]
        assert b["qualifies"] is False
        assert b["exclusion_label"] == "no on-screen text of any kind"
