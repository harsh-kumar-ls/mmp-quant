"""ROUGE-L tests, with the expected values worked out by hand in the comments.

ROUGE-L is implemented in-repo rather than pulled from a library so that the
tokenisation matches our normalisation exactly and the metric is auditable for
the thesis. That means it needs its own tests against hand-computed values.
"""

from __future__ import annotations

import pytest

from mmpquant.overlap import clamp_similarity, lcs_length, rouge_l_f1


class TestLcsLength:
    def test_identical(self):
        assert lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_disjoint(self):
        assert lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_empty_either_side(self):
        assert lcs_length([], ["a"]) == 0
        assert lcs_length(["a"], []) == 0

    def test_subsequence_need_not_be_contiguous(self):
        # "a c e" appears in order inside the second sequence.
        assert lcs_length(["a", "c", "e"], ["a", "b", "c", "d", "e"]) == 3

    def test_order_matters(self):
        # Same multiset, reversed order: LCS is 1, not 3. This is exactly why
        # OCR lines must be sorted into reading order before comparison.
        assert lcs_length(["a", "b", "c"], ["c", "b", "a"]) == 1

    def test_symmetric(self):
        a, b = ["x", "y", "z", "y"], ["y", "z", "w"]
        assert lcs_length(a, b) == lcs_length(b, a)

    def test_repeated_tokens_counted_once_per_position(self):
        assert lcs_length(["a", "a", "a"], ["a", "a"]) == 2


class TestRougeLF1:
    def test_identical_text_is_one(self):
        text = "mitochondria generate ATP through oxidative phosphorylation"
        assert rouge_l_f1(text, text) == pytest.approx(1.0)

    def test_hand_computed_partial_overlap(self):
        # a = [the, cat, sat, on, the, mat]      (6 tokens)
        # b = [the, dog, stood, on, a, hill]     (6 tokens)
        # LCS = [the, on] -> 2
        # P = 2/6, R = 2/6, F1 = 2/6 = 0.3333...
        got = rouge_l_f1("the cat sat on the mat", "the dog stood on a hill")
        assert got == pytest.approx(2 / 6, abs=1e-6)

    def test_hand_computed_asymmetric_lengths(self):
        # a = [atp]                              (1 token)
        # b = [atp, is, the, energy, currency]   (5 tokens)
        # LCS = 1 ; P = 1/1, R = 1/5
        # F1 = 2*1*0.2/(1.2) = 0.3333...
        got = rouge_l_f1("ATP", "ATP is the energy currency")
        assert got == pytest.approx(1 / 3, abs=1e-6)

    def test_no_overlap_is_zero(self):
        assert rouge_l_f1("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_empty_side_is_zero_not_an_error(self):
        # A text-free frame must yield 0 overlap, and must never raise — the
        # scorer relies on this to keep going through blank windows.
        assert rouge_l_f1("", "anything at all") == 0.0
        assert rouge_l_f1("anything at all", "") == 0.0
        assert rouge_l_f1("", "") == 0.0

    def test_punctuation_only_side_is_zero(self):
        assert rouge_l_f1("•  —", "real narration here") == 0.0

    def test_f1_is_symmetric(self):
        a, b = "the cat sat on the mat", "the dog stood on a hill"
        assert rouge_l_f1(a, b) == pytest.approx(rouge_l_f1(b, a))

    def test_normalisation_is_applied(self):
        # Same words, different renderings on each side. Without normalisation
        # this would score well below 1.0 and under-report real redundancy.
        assert rouge_l_f1("PHOTO-\nSYNTHESIS HAS 3 STAGES",
                          "photosynthesis has three stages") == pytest.approx(1.0)

    def test_word_order_scramble_scores_below_identical(self):
        forward = "energy is released from the bond"
        reversed_ = "bond the from released is energy"
        assert rouge_l_f1(forward, reversed_) < rouge_l_f1(forward, forward)

    def test_range_is_zero_to_one(self):
        pairs = [
            ("a b c d e", "a b c d e"),
            ("a b c d e", "f g h i j"),
            ("a b c", "a x b x c"),
            ("", "a"),
        ]
        for a, b in pairs:
            assert 0.0 <= rouge_l_f1(a, b) <= 1.0


class TestClampSimilarity:
    def test_negative_cosine_becomes_zero(self):
        # Cosine can go negative; a negative "overlap" is not meaningful and
        # would drag the weighted score below the documented 0-100 range.
        assert clamp_similarity(-0.4) == 0.0

    def test_above_one_is_capped(self):
        assert clamp_similarity(1.0000001) == 1.0

    def test_passes_through_valid_range(self):
        assert clamp_similarity(0.37) == pytest.approx(0.37)
