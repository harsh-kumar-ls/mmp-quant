"""Normalisation tests.

Each case here is a real way OCR and ASR disagree about the *same* words. If
normalisation regresses, verbatim overlap silently under-reports redundancy, so
these are correctness tests for the score, not string-utility trivia.
"""

from __future__ import annotations

from mmpquant.textnorm import (
    NormalisationConfig,
    normalise_for_semantic,
    normalise_for_verbatim,
    tokenise,
    word_count,
)


class TestVerbatimNormalisation:
    def test_lowercases_slide_shouting(self):
        # Slides use all-caps headings; ASR does not.
        assert normalise_for_verbatim("MITOCHONDRIA") == "mitochondria"

    def test_rejoins_hyphenated_linebreak(self):
        # OCR reads a wrapped word as two pieces; ASR emits one word.
        assert normalise_for_verbatim("photo-\nsynthesis") == "photosynthesis"

    def test_rejoins_unicode_hyphen_variants(self):
        # PP-OCR frequently returns en-dashes where the slide had a hyphen.
        assert normalise_for_verbatim("photo–\nsynthesis") == "photosynthesis"

    def test_plain_newline_becomes_space_not_join(self):
        # A newline without a hyphen separates two words; joining them would
        # invent a token that matches nothing.
        assert normalise_for_verbatim("electron\ntransport") == "electron transport"

    def test_strips_bullets_and_punctuation(self):
        assert normalise_for_verbatim("• ATP, — and NADH.") == "atp and nadh"

    def test_number_words_become_digits(self):
        # Slide: "3 stages". Narrator: "three stages".
        assert normalise_for_verbatim("3 stages") == normalise_for_verbatim("three stages")

    def test_number_normalisation_is_two_way_consistent(self):
        assert normalise_for_verbatim("Step Two") == "step 2"

    def test_collapses_runs_of_whitespace(self):
        assert normalise_for_verbatim("a    b\t\tc") == "a b c"

    def test_nfkc_folds_ligatures(self):
        # U+FB01 'fi' ligature is common in OCR of serif slide fonts.
        assert normalise_for_verbatim("deﬁnition") == "definition"

    def test_config_flags_are_respected(self):
        cfg = NormalisationConfig(lowercase=False, words_to_digits=False)
        assert normalise_for_verbatim("Three ATP", cfg) == "Three ATP"

    def test_empty_and_punctuation_only_are_empty(self):
        assert normalise_for_verbatim("") == ""
        assert normalise_for_verbatim("•  —  ...") == ""


class TestSemanticNormalisation:
    def test_preserves_case_and_punctuation(self):
        # The embedding model is trained on natural text; stripping case and
        # punctuation degrades its vectors.
        text = "Mitochondria produce ATP. Why does that matter?"
        assert normalise_for_semantic(text) == text

    def test_still_fixes_ocr_line_wrapping(self):
        assert normalise_for_semantic("photo-\nsynthesis is key") == "photosynthesis is key"

    def test_flattens_newlines(self):
        assert normalise_for_semantic("Line one\nLine two") == "Line one Line two"


class TestWordCount:
    def test_counts_words_not_glyphs(self):
        # This is the input to the signaling-exception threshold: bullets and
        # dashes must not inflate a 2-word label past min_onscreen_words.
        assert word_count("• ATP — synthase") == 2

    def test_short_label_stays_below_default_threshold_of_five(self):
        assert word_count("Key Terms:") < 5

    def test_sentence_exceeds_threshold(self):
        assert word_count("Mitochondria generate ATP through oxidative phosphorylation") >= 5

    def test_empty_is_zero(self):
        assert word_count("") == 0
        assert word_count("\n\n") == 0


def test_tokenise_splits_on_whitespace():
    assert tokenise("a b  c") == ["a", "b", "c"]
