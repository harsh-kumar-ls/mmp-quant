"""Defect 10.3 (04 Oct 2026): OCR misreads of chalk handwriting scored as prose.

Test_22 is a chalkboard lecture filmed with the lecturer moving in front of the
board. OCR returned confident garbage ("electmmoquehisin" for
"electromagnetism"), and spaCy -- which must tag every token as something --
tagged several garbage tokens as finite verbs. Garbage became a "sentence" and
scored 11.18 for a stretch of video containing no readable sentence at all.

Every fixture below is real OCR output from the corpus, not invented.
"""

from __future__ import annotations

import json

import pytest

from mmpquant.extract.ocr.base import RELIABILITY_OK, assess_reading_quality
from mmpquant.onscreen import (
    KIND_LABEL,
    KIND_SENTENCE,
    classify_text,
    dedupe_lines,
    is_real_word,
    real_word_share,
)
from mmpquant.stages import DEDUP_WARNING_PREFIX, dedup_warning, load_ocr
from mmpquant.types import OcrBlock, OcrLine


class TestRealWords:
    @pytest.mark.parametrize("word", [
        "boils", "is", "electromagnetism", "eigenvalue", "photosynthesis", "determinant",
    ])
    def test_real_words_including_technical_ones(self, word):
        assert is_real_word(word)

    @pytest.mark.parametrize("word", [
        "moguehiin", "electmmoquehisin", "quantwnmuchanics", "lengfh", "mathcmahiccl",
    ])
    def test_corpus_ocr_garbage(self, word):
        assert not is_real_word(word)

    def test_share_skips_short_tokens_and_notation(self):
        # "x", "A", "3" are as likely notation as noise; only 3+ letter words count.
        assert real_word_share("x = 3A + boils") == (1, 1)


class TestGarbageVerbsNoLongerMakeSentences:
    @pytest.mark.parametrize("text", [
        "mechanies\nmoguehiin\nmchanics\nphysics",                 # tagged 'moguehiin' Fin
        "proper\nClassieaQ mechanies\nelectmmoquehisin\nmelhami(s",  # 'electmmoquehisin'
        "classical\nelectmmoq\nwwon\nStxhshc",                     # 'wwon'
        "Lana\nLemy\nlemgf\nlengfh\nlenyfh\nlenyfh\nm",             # Test_20 chalkboard
    ])
    def test_test_22_and_test_20_garbage_is_a_label(self, text):
        kind, evidence = classify_text(text)
        assert kind == KIND_LABEL
        assert evidence["finite_verbs"] == []

    def test_a_real_sentence_with_misread_content_words_is_kept(self):
        # Test_14JzbhQX00: "Subspace S is orthogonal to Subspace" with
        # "orthogonal" misread. The verb is real, so it is still a sentence --
        # and must stay one. An element-level "mostly garbage -> label" gate was
        # tried and removed because it threw exactly these away, hiding real
        # redundancy (the flattering direction). Do not reintroduce it without
        # re-checking this case.
        kind, evidence = classify_text("Sulspace S is orlefoul to Subspace")
        assert kind == KIND_SENTENCE
        assert evidence["finite_verbs"] == ["is"]


class TestContractions:
    def test_contracted_finite_verb_counts(self):
        # Test_12. The "'s" was always rejected for not being alphabetic; this
        # caption only ever counted because noise ("πTT") stood in for the verb.
        kind, evidence = classify_text("But that's a super\nspecific case")
        assert kind == KIND_SENTENCE
        assert "'s" in evidence["finite_verbs"]

    def test_curly_apostrophe_too(self):
        kind, _ = classify_text("But that’s a super specific case")
        assert kind == KIND_SENTENCE

    def test_possessive_is_not_a_verb(self):
        # "'s" tagged PART (possessive), not VerbForm=Fin, so it never reaches
        # the contraction allowance.
        kind, _ = classify_text("Ohm's Law")
        assert kind == KIND_LABEL


class TestRepeatReadingsKeepTheLongerOne:
    def test_short_label_does_not_delete_the_sentence_it_is_a_substring_of(self):
        # Test_11/13/14: "Area" sat above "areas get scaled by det(A)", and the
        # caption was discarded as a repeat of the label before being tagged.
        out = dedupe_lines("Area\ndet(A)\nareas get scaled by detA")
        assert "areas get scaled by detA" in out
        kind, _ = classify_text("Area\ndet(A)\nareas get scaled by detA")
        assert kind == KIND_SENTENCE

    def test_partial_renders_still_collapse_to_the_fullest(self):
        out = dedupe_lines("elementary\nelementary particles\nelementary")
        assert out == "elementary particles"

    def test_exact_repeats_collapse(self):
        assert dedupe_lines("Water boils\nWater boils") == "Water boils"


class TestVideoLevelShareIsRecordedNotAVerdict:
    """Measured on all 24 videos, the video-level real-word share does not
    separate misread videos from clean ones (missed Test_20; falsely flagged
    test_1, whose OCR was correct but dropped spaces, and 14_JzbhQX00, whose
    dominant text is a watermark). It is recorded, never used as a verdict."""

    def block(self, text, start=0.0, end=10.0):
        return OcrBlock(start=start, end=end, lines=(OcrLine(text, 0.9, None),),
                        source_frame_time=start, n_frames_collapsed=1)

    def test_share_is_recorded(self):
        rel = assess_reading_quality({"reliability": RELIABILITY_OK},
                                     [self.block("water boils quickly")])
        assert rel["real_word_share"] == 1.0

    def test_garbage_never_changes_the_verdict(self):
        rel = assess_reading_quality({"reliability": RELIABILITY_OK},
                                     [self.block("moguehiin electmmoquehisin lengfh")])
        assert rel["real_word_share"] == 0.0
        assert rel["reliability"] == RELIABILITY_OK

    def test_share_is_weighted_by_time_on_screen(self):
        # 90 s of real text, 10 s of garbage: duration, not block count (9.1).
        rel = assess_reading_quality(
            {"reliability": RELIABILITY_OK},
            [self.block("water boils quickly", 0, 90), self.block("moguehiin lengfh electmmoquehisin", 90, 100)],
        )
        assert rel["real_word_share"] == pytest.approx(0.9, abs=0.01)


class TestDedupWarningIsHonest:
    def test_no_longer_claims_it_cannot_affect_correctness(self):
        w = dedup_warning(0.016)
        assert "not a correctness problem" not in w
        assert "1.6%" in w
        assert "inspect" in w

    def test_cached_runs_get_the_corrected_wording(self, tmp_path):
        old = (f"{DEDUP_WARNING_PREFIX} (1.6% of frames collapsed) — ... "
               "This is a cost signal, not a correctness problem.")
        path = tmp_path / "ocr.json"
        path.write_text(json.dumps({"dedup_ratio": 0.016, "warnings": [old, "other"],
                                    "blocks": []}))
        _, warnings = load_ocr(path)
        assert "not a correctness problem" not in warnings[0]
        assert warnings[1] == "other"


class TestOnscreenConfigIsActuallyRead:
    """Before 04 Oct 2026, `onscreen.line_gap_ratio`, `x_align_tol_px` and
    `dedup_similarity` were documented in config.yaml but never read: the code
    used hard-coded defaults that happened to match, so editing them did
    nothing."""

    def test_every_onscreen_value_reaches_the_params(self):
        from mmpquant.config import Config
        from mmpquant.stages import _onscreen_params

        cfg = Config.load("config.yaml")
        raw = json.loads(json.dumps(cfg.raw))
        raw["onscreen"].update(line_gap_ratio=9.0, x_align_tol_px=99.0, dedup_similarity=0.5)
        p = _onscreen_params(Config(raw=raw, source_path=cfg.source_path))
        assert (p.line_gap_ratio, p.x_align_tol_px, p.dedup_similarity) == (9.0, 99.0, 0.5)
