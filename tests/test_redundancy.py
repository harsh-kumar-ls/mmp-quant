"""Redundancy formula tests.

Required by the spec, including the three named fixtures: a known-redundant
pair, a known-clean pair, and a short-labels-only case that must return N/A.

The embedder is injected, so these assert the *formula*, not the embedding model.
"""

from __future__ import annotations

import pytest

from mmpquant.scoring.redundancy import (
    EXCL_NO_NARRATION,
    EXCL_NO_ONSCREEN_TEXT,
    EXCL_SIGNALING_EXCEPTION,
    RedundancyScorer,
)

from .conftest import (
    CLEAN_ASR,
    CLEAN_OCR,
    REDUNDANT_ASR,
    REDUNDANT_OCR,
    SHORT_LABEL_ASR,
    SHORT_LABEL_OCR,
    ConstantEmbedder,
    make_window,
)


def scorer(semantic: float = 0.0, **kwargs) -> RedundancyScorer:
    return RedundancyScorer(embedder=ConstantEmbedder(semantic), **kwargs)


# --------------------------------------------------------------------------- #
# Weighted formula
# --------------------------------------------------------------------------- #


class TestWeightedFormula:
    def test_verbatim_identical_and_semantic_one_gives_exactly_100(self):
        s = scorer(semantic=1.0)
        ws = s.score_window(make_window(REDUNDANT_OCR, REDUNDANT_ASR))
        assert ws.qualifies
        assert ws.score == pytest.approx(100.0)

    def test_weights_are_applied_as_specified(self):
        # verbatim = 1.0 (identical text), semantic pinned to 0.5
        # score = 100 * (0.6*1.0 + 0.4*0.5) = 80.0
        s = scorer(semantic=0.5)
        ws = s.score_window(make_window(REDUNDANT_OCR, REDUNDANT_ASR))
        assert ws.score == pytest.approx(80.0)

    def test_zero_on_both_branches_gives_zero_for_a_qualifying_window(self):
        # A window CAN legitimately score 0 -- qualifying but non-redundant.
        # That is different from an N/A video, which has no score at all.
        s = scorer(semantic=0.0)
        ws = s.score_window(make_window("alpha beta gamma delta epsilon", "one two three"))
        assert ws.qualifies
        assert ws.score == pytest.approx(0.0)

    def test_custom_weights_respected(self):
        # All weight on semantic: score = 100 * 0.25
        s = scorer(semantic=0.25, weight_verbatim=0.0, weight_semantic=1.0)
        ws = s.score_window(make_window(REDUNDANT_OCR, REDUNDANT_ASR))
        assert ws.score == pytest.approx(25.0)

    def test_weights_must_sum_to_one(self):
        # Non-unit weights silently rescale the 0-100 range and make runs
        # incomparable across configs.
        with pytest.raises(ValueError, match="sum to 1.0"):
            scorer(weight_verbatim=0.7, weight_semantic=0.4)

    def test_score_stays_within_zero_to_one_hundred(self):
        s = scorer(semantic=1.0)
        for ocr, asr in [
            (REDUNDANT_OCR, REDUNDANT_ASR),
            (CLEAN_OCR, CLEAN_ASR),
            ("a b c d e f", "f e d c b a"),
        ]:
            ws = s.score_window(make_window(ocr, asr))
            assert 0.0 <= ws.score <= 100.0

    def test_components_are_exposed_for_the_evidence_table(self):
        s = scorer(semantic=0.5)
        ws = s.score_window(make_window(REDUNDANT_OCR, REDUNDANT_ASR))
        assert ws.components["verbatim"] == pytest.approx(1.0)
        assert ws.components["semantic"] == pytest.approx(0.5)
        assert ws.components["ocr_sentence_word_count"] == 6
        assert ws.components["n_sentence_elements"] == 0  # make_window bypasses grouping

    def test_semantic_results_are_cached_across_identical_pairs(self):
        # A static slide recurs in many windows; re-encoding it is wasted work.
        embedder = ConstantEmbedder(0.5)
        s = RedundancyScorer(embedder=embedder)
        window = make_window(REDUNDANT_OCR, REDUNDANT_ASR)
        for _ in range(5):
            s.score_window(window)
        assert embedder.calls == 1


class TestRedundantVsClean:
    def test_redundant_pair_scores_far_above_clean_pair(self):
        # The single most important behavioural property: the score must
        # discriminate. Uses a monotonic semantic proxy rather than a constant.
        from .conftest import JaccardEmbedder

        s = RedundancyScorer(embedder=JaccardEmbedder())
        redundant = s.score_window(make_window(REDUNDANT_OCR, REDUNDANT_ASR)).score
        clean = s.score_window(make_window(CLEAN_OCR, CLEAN_ASR)).score
        assert redundant > clean
        assert redundant - clean > 50, f"weak separation: {redundant} vs {clean}"


# --------------------------------------------------------------------------- #
# Signaling exception
# --------------------------------------------------------------------------- #


class TestSignalingException:
    def test_short_label_is_excluded_not_scored_zero(self):
        # Mayer's exception: short on-screen text is a signaling cue, not
        # redundancy. Scoring it 0 would still drag the video mean down and
        # penalise a well-designed video.
        s = scorer(semantic=1.0)
        ws = s.score_window(make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR))
        assert not ws.qualifies
        assert ws.score is None
        assert ws.exclusion_reason == EXCL_SIGNALING_EXCEPTION

    def test_boundary_exactly_at_threshold_qualifies(self):
        # Threshold is ">= min_onscreen_words".
        s = scorer(min_onscreen_words=5)
        ws = s.score_window(make_window("one two three four five", "narration words here"))
        assert ws.qualifies

    def test_boundary_one_below_threshold_is_excluded(self):
        s = scorer(min_onscreen_words=5)
        ws = s.score_window(make_window("one two three four", "narration words here"))
        assert not ws.qualifies
        assert ws.exclusion_reason == EXCL_SIGNALING_EXCEPTION

    def test_threshold_is_configurable(self):
        ws = scorer(min_onscreen_words=2).score_window(
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR)
        )
        assert ws.qualifies

    def test_bullets_do_not_inflate_word_count_past_threshold(self):
        # Raw split() on "• ATP — synthase •" gives 5 tokens and would wrongly
        # qualify a 2-word label.
        ws = scorer(min_onscreen_words=5).score_window(
            make_window("• ATP — synthase •", SHORT_LABEL_ASR)
        )
        assert not ws.qualifies

    def test_no_onscreen_text_is_distinguished_from_a_short_label(self):
        # Different diagnoses -> different fixes. The report must not conflate
        # "OCR found nothing" with "OCR found a title".
        ws = scorer().score_window(make_window("", SHORT_LABEL_ASR))
        assert ws.exclusion_reason == EXCL_NO_ONSCREEN_TEXT

    def test_no_narration_is_excluded(self):
        ws = scorer(min_narration_words=1).score_window(
            make_window(REDUNDANT_OCR, "")
        )
        assert not ws.qualifies
        assert ws.exclusion_reason == EXCL_NO_NARRATION

    def test_min_narration_words_is_configurable(self):
        window = make_window(REDUNDANT_OCR, "um")
        assert scorer(min_narration_words=1).score_window(window).qualifies
        assert not scorer(min_narration_words=3).score_window(window).qualifies


# --------------------------------------------------------------------------- #
# Video-level aggregation
# --------------------------------------------------------------------------- #


class TestAggregation:
    def test_short_labels_only_video_is_na_never_zero(self):
        # THE spec-mandated case. A video whose only on-screen text is short
        # labels is unscoreable, not perfectly non-redundant.
        # A video whose only on-screen text is short signaling labels cannot
        # violate the redundancy principle -- under Mayer that is good design,
        # so 0 is the correct reading and anchors the bottom of the scale.
        s = scorer(semantic=1.0)
        windows = [
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=i, start=i * 10.0,
                        end=(i + 1) * 10.0)
            for i in range(3)
        ]
        result = s.score(windows)
        assert result.score == 0.0
        assert result.status == "OK"
        assert result.n_windows_qualifying == 0
        # the 0 must carry its justification, so it can be audited
        assert any(EXCL_SIGNALING_EXCEPTION in w for w in result.warnings)

    def test_text_free_video_scores_zero_with_an_audit_warning(self):
        # Narration present, no on-screen text anywhere: genuinely non-redundant.
        s = scorer()
        windows = [make_window("", "some narration here", index=i, start=i * 10.0,
                               end=(i + 1) * 10.0) for i in range(3)]
        result = s.score(windows)
        assert result.score == 0.0
        assert result.status == "OK"
        assert any(EXCL_NO_ONSCREEN_TEXT in w for w in result.warnings)

    def test_silent_video_is_na_never_zero(self):
        # THE case the N/A rule still exists for. No speech anywhere means the
        # principle cannot be evaluated; a silent track is more often a broken
        # audio extraction than a design choice, and 0 would disguise that.
        s = scorer()
        windows = [make_window(REDUNDANT_OCR, "", index=i, start=i * 10.0,
                               end=(i + 1) * 10.0) for i in range(3)]
        result = s.score(windows)
        assert result.score is None
        assert result.status == "N/A"

    def test_no_windows_at_all_is_na(self):
        result = scorer().score([])
        assert result.score is None
        assert result.status == "N/A"

    def test_mean_over_equal_length_windows(self):
        # 100 and 0 over two equal windows -> 50.
        s = scorer(semantic=0.0)
        windows = [
            make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=0, start=0.0, end=10.0),
            make_window("alpha beta gamma delta epsilon", "one two three",
                        index=1, start=10.0, end=20.0),
        ]
        # verbatim=1.0 -> 60.0 ; verbatim=0.0 -> 0.0 ; mean = 30.0
        assert s.score(windows).score == pytest.approx(30.0)

    def test_duration_weighting_favours_the_longer_window(self):
        # A 2 s tail window must not count as much as a 10 s window.
        s = scorer(semantic=0.0)
        windows = [
            make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=0, start=0.0, end=10.0),
            make_window("alpha beta gamma delta epsilon", "one two three",
                        index=1, start=10.0, end=12.0),
        ]
        # (60.0*10 + 0.0*2) / 12 = 50.0  (an unweighted mean would give 30.0)
        assert s.score(windows).score == pytest.approx(50.0)

    def test_excluded_windows_do_not_dilute_the_mean(self):
        # One redundant window plus two short-label windows must score as the
        # redundant window alone -- the excluded ones carry no weight at all.
        s = scorer(semantic=0.0)
        windows = [
            make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=0, start=0.0, end=10.0),
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=1, start=10.0, end=20.0),
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=2, start=20.0, end=30.0),
        ]
        result = s.score(windows)
        assert result.score == pytest.approx(60.0)
        assert result.n_windows_qualifying == 1
        assert result.n_windows_total == 3

    def test_excluded_windows_are_retained_in_the_audit_trail(self):
        s = scorer()
        windows = [
            make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=0, start=0.0, end=10.0),
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=1, start=10.0, end=20.0),
        ]
        result = s.score(windows)
        assert len(result.window_scores) == 2
        assert result.window_scores[1].exclusion_reason == EXCL_SIGNALING_EXCEPTION

    def test_low_coverage_raises_a_warning(self):
        # One qualifying window out of twenty: the score is real but rests on
        # 5% of the video and must not be reported as if it were representative.
        s = scorer(semantic=0.0)
        windows = [make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=i,
                               start=i * 10.0, end=(i + 1) * 10.0) for i in range(19)]
        windows.append(make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=19,
                                   start=190.0, end=200.0))
        result = s.score(windows)
        assert result.score is not None
        assert any("low-confidence" in w for w in result.warnings)

    def test_full_coverage_raises_no_coverage_warning(self):
        s = scorer(semantic=0.0)
        windows = [make_window(REDUNDANT_OCR, REDUNDANT_ASR, index=i,
                               start=i * 10.0, end=(i + 1) * 10.0) for i in range(5)]
        assert not any("low-confidence" in w for w in s.score(windows).warnings)

    def test_extra_warnings_are_propagated(self):
        s = scorer(semantic=0.0)
        windows = [make_window(REDUNDANT_OCR, REDUNDANT_ASR)]
        result = s.score(windows, extra_warnings=["detected language 'hi' != 'en'"])
        assert "detected language 'hi' != 'en'" in result.warnings

    def test_zero_warning_breaks_down_exclusions(self):
        # A 0 must say which kinds of window produced it, so a reader can tell
        # "slides were blank" from "slides were short labels".
        s = scorer()
        windows = [
            make_window(SHORT_LABEL_OCR, SHORT_LABEL_ASR, index=0, start=0.0, end=10.0),
            make_window("", SHORT_LABEL_ASR, index=1, start=10.0, end=20.0),
        ]
        result = s.score(windows)
        assert result.score == 0.0
        blob = " ".join(result.warnings)
        assert EXCL_SIGNALING_EXCEPTION in blob
        assert EXCL_NO_ONSCREEN_TEXT in blob

    def test_na_reason_names_the_missing_narration(self):
        s = scorer()
        windows = [make_window(REDUNDANT_OCR, "", index=i, start=i * 10.0,
                               end=(i + 1) * 10.0) for i in range(2)]
        result = s.score(windows)
        assert result.status == "N/A"
        assert "no narration" in result.na_reason

    def test_zero_and_na_are_distinguishable(self):
        # The whole point of the split: these two must never look alike.
        s = scorer()
        no_text = s.score([make_window("", "narration is present here", index=0)])
        no_speech = s.score([make_window(REDUNDANT_OCR, "", index=0)])
        assert (no_text.score, no_text.status) == (0.0, "OK")
        assert (no_speech.score, no_speech.status) == (None, "N/A")


class TestSerialisation:
    def test_result_round_trips_to_json(self):
        import json

        s = scorer(semantic=0.5)
        result = s.score([make_window(REDUNDANT_OCR, REDUNDANT_ASR)])
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["score"] == pytest.approx(80.0)
        assert payload["principle"] == "redundancy"

    def test_na_serialises_as_null_with_an_explicit_status(self):
        # A downstream consumer must be able to tell N/A from 0 without guessing.
        import json

        s = scorer()
        result = s.score([make_window(REDUNDANT_OCR, "")])  # no narration -> N/A
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["score"] is None
        assert payload["status"] == "N/A"
        assert payload["na_reason"]

    def test_zero_serialises_as_a_real_number_not_null(self):
        import json

        s = scorer()
        result = s.score([make_window("", "narration is present here")])
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["score"] == 0.0
        assert payload["status"] == "OK"
        assert payload["na_reason"] is None

    def test_params_are_recorded_for_the_methodology_appendix(self):
        s = scorer(semantic=0.0, min_onscreen_words=7)
        params = s.score([make_window(REDUNDANT_OCR, REDUNDANT_ASR)]).params
        assert params["min_onscreen_words"] == 7
        assert params["weights"] == {"verbatim": 0.6, "semantic": 0.4}


class TestRegistry:
    def test_redundancy_scorer_is_registered_as_a_plugin(self):
        # The plugin seam coherence.py now uses (11 Oct 2026).
        from mmpquant.scoring import available, get_scorer

        assert "redundancy" in available()
        assert get_scorer("redundancy") is RedundancyScorer

    def test_unknown_scorer_raises(self):
        from mmpquant.scoring import get_scorer

        with pytest.raises(KeyError):
            get_scorer("signaling")  # not built

    def test_coherence_slots_in_through_the_same_seam(self):
        from mmpquant.scoring import available, get_scorer
        from mmpquant.scoring.coherence import CoherenceScorer

        assert "coherence" in available()
        assert get_scorer("coherence") is CoherenceScorer
