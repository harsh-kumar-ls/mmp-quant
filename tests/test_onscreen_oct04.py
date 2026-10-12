"""On-screen text fixes of 04 Oct 2026: goal statements, word splitting,
per-moment grouping, majority vote across readings, Apple box conversion,
and the zero-length final segment. Fixture strings are real corpus OCR."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmpquant.extract.align import OnscreenParams, _elements_for_window
from mmpquant.extract.ocr.apple import _to_pixels
from mmpquant.onscreen import (
    KIND_LABEL,
    KIND_SENTENCE,
    TextElement,
    classify_text,
    collapse_near_duplicates,
)
from mmpquant.segments import segment_bounds
from mmpquant.types import OcrBlock, OcrLine
from mmpquant.wordsplit import split_run_together


class TestGoalStatementsAreTitles:
    """User's ruling: a goal statement orients the learner like a title and is
    never redundancy, even read aloud."""

    @pytest.mark.parametrize("text", [
        "Aim: Provide proper mathematical language for classical mechanics",
        "Learning objectives: Students will understand dilution",   # finite verb, still a goal
        "Goals - we compute the determinant",
        "OUTCOME: You can solve dilution problems",
    ])
    def test_goal_headings(self, text):
        kind, evidence = classify_text(text)
        assert kind == KIND_LABEL
        assert evidence["reason"] == "goal_statement"

    def test_a_sentence_about_an_aim_is_still_a_sentence(self):
        kind, _ = classify_text("The aim of chemistry is understanding matter")
        assert kind == KIND_SENTENCE


class TestWordSplitting:
    @pytest.mark.parametrize("glued,expected", [
        ("Heatcannotspontaneouslyflowfromacolderbody",
         "Heat cannot spontaneously flow from a colder body"),
        ("Clausiusstatementofthesecondlawofthermodynamics",
         "Clausius statement of the second law of thermodynamics"),
        ("HeatTransfer", "Heat Transfer"),
        ("thisvideo", "this video"),
    ])
    def test_run_together_english_is_split(self, glued, expected):
        assert split_run_together(glued) == expected

    @pytest.mark.parametrize("garbage", [
        "moguehiin", "electmmoquehisin", "quantwnmuchanics", "orlefoul", "Caryage",
        "clideo", "NPTEL", "fassicalmechauies", "Leucophaeamaderaeandthe",
    ])
    def test_garbage_and_names_are_left_alone(self, garbage):
        assert split_run_together(garbage) == garbage

    def test_one_misread_word_is_not_split_into_a_fake_verb(self):
        # Apple OCR read "electromagnetism" on chalk as "electromagnetisin";
        # splitting gave "electromagnet is in" and a finite verb out of nothing.
        assert split_run_together("electromagnetisin") == "electromagnetisin"

    def test_split_makes_the_hidden_verb_visible(self):
        kind, _ = classify_text(split_run_together("Heatcannotspontaneouslyflowfromacolderbody"))
        assert kind == KIND_SENTENCE

    def test_equations_and_punctuation_untouched(self):
        assert split_run_together("det(A) = 3c + 2y") == "det(A) = 3c + 2y"


def line(text, x0, y0, x1, y1):
    return OcrLine(text=text, confidence=0.95, bbox=(x0, y0, x1, y1))


class TestOnlyTextOnScreenTogetherFormsAnElement:
    def test_test_11_title_and_later_label_do_not_merge(self):
        # Test_11: "Gaussian Elimination" (shown first) and "Signed Area =
        # det(A)y" (shown seconds later, just below) were pooled into one
        # element that parsed as "Gaussian Elimination *signed* Area".
        early = OcrBlock(0.0, 4.0, (line("Gaussian Elimination", 381, 66, 900, 120),), 0.0, 1)
        late = OcrBlock(4.0, 10.0, (line("Signed Area = det(A)y", 381, 140, 900, 216),), 4.0, 1)
        elements = _elements_for_window((early, late), [0, 1], OnscreenParams())
        texts = sorted(e.text for e in elements)
        assert texts == ["Gaussian Elimination", "Signed Area = det(A)y"]
        assert all(e.kind == KIND_LABEL for e in elements)


def element(text, kind):
    return TextElement(lines=(line(text, 0, 0, 100, 20),), kind=kind, bbox=(0, 0, 100, 20))


class TestMajorityOfReadings:
    def test_one_misread_in_many_does_not_make_a_title_a_sentence(self):
        # Test_22's title card, read many times; one reading has "fer" for
        # "for", tagged as a verb.
        good = "GEOMETRIC ANATOMY OF THEORETICAL PHYSICS Frederic P. Schuller, Institute for Physics"
        bad = "GEOMETRIC ANATOMY OF THEORETICAL PHYSICS Frederie P. Schuller, Institute fer Physics"
        out = collapse_near_duplicates([element(good, KIND_LABEL)] * 5 + [element(bad, KIND_SENTENCE)])
        assert len(out) == 1
        assert out[0].kind == KIND_LABEL

    def test_a_consistently_read_sentence_stays_a_sentence(self):
        s = "All areas get scaled by the determinant of the matrix"
        out = collapse_near_duplicates([element(s, KIND_SENTENCE)] * 4 + [element(s + "x", KIND_LABEL)])
        assert out[0].kind == KIND_SENTENCE

    def test_a_short_label_inside_a_sentence_does_not_vote(self):
        out = collapse_near_duplicates([
            element("areas get scaled by det(A)", KIND_SENTENCE), element("Area", KIND_LABEL)])
        assert [e.kind for e in out] == [KIND_SENTENCE]


class TestAppleBoxes:
    def test_bottom_left_normalised_box_becomes_top_left_pixels(self):
        rect = SimpleNamespace(origin=SimpleNamespace(x=0.1, y=0.7),
                               size=SimpleNamespace(width=0.5, height=0.2))
        x0, y0, x1, y1 = _to_pixels(rect, 1000, 500)
        assert (x0, x1) == pytest.approx((100, 600))
        assert (y0, y1) == pytest.approx((50, 150))  # near the TOP of the image


class TestSegmentTail:
    def test_a_sub_tenth_second_tail_folds_into_the_last_segment(self):
        # test9 is 180.098 s; it used to get a 0.098 s seventh segment.
        bounds = segment_bounds(180.098, 30.0)
        assert len(bounds) == 6
        assert bounds[-1] == pytest.approx((150.0, 180.098))

    def test_a_real_short_tail_is_kept(self):
        assert len(segment_bounds(185.0, 30.0)) == 7


class TestHeadingsAreNotSentences:
    """05 Oct 2026: headings whose nouns were mis-tagged as verbs, from the
    corpus. Fixed by the transformer parser plus the subject requirement."""

    @pytest.mark.parametrize("text", [
        "Error correction codes\nHamming codes",          # test6: "codes" as a verb
        "SET THEORY\nLOGIC\nOHI\nDAT",                    # Test_23: "set", "dat"
        "Gaussian Elimination\nSigned Area =det(A)y",     # Test_11: "Signed"
    ])
    def test_heading_is_a_label(self, text):
        assert classify_text(text)[0] == KIND_LABEL

    @pytest.mark.parametrize("text", [
        "Mitochondria generate ATP through\noxidative phosphorylation",
        "This machine flips the second\nqubit if acting on the first\nqubit is 1",
        "The concentration of a 300mL aqueous KCl solution\nwas increased from 0.25M to 0.75M",
    ])
    def test_wrapped_sentence_is_a_sentence(self, text):
        assert classify_text(text)[0] == KIND_SENTENCE


class TestDecodingLanguageWarning:
    """05 Oct 2026: transcription is fixed to English but detection still runs,
    so a mismatch is reported rather than hidden."""

    def transcript(self, lang, p):
        from mmpquant.types import AsrSegment, AsrWord, Transcript
        seg = AsrSegment(0.0, 1.0, "hello there", (AsrWord(0.0, 1.0, "hello"),))
        return Transcript((seg,), lang, p, 1.0)

    def test_mismatch_with_the_decoding_language_is_reported(self):
        from mmpquant.extract.asr import language_warnings
        w = language_warnings(self.transcript("cy", 0.8), "en", 0.5, decoded_as="en")
        assert any("transcribed as 'en'" in x and "'cy'" in x for x in w)

    def test_agreement_is_silent(self):
        from mmpquant.extract.asr import language_warnings
        assert language_warnings(self.transcript("en", 0.99), "en", 0.5, decoded_as="en") == []
