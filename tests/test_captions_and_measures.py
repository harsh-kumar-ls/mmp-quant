"""15 Oct 2026: captions that run one sentence across screens, and the two
redundancy measures changed to compare like with like (user decisions,
prompted by Microlearning-Safety-Training: read aloud word for word, scored
41.55 before)."""

from __future__ import annotations

import pytest

from mmpquant.extract.captions import caption_stream, stitch_sentences
from mmpquant.overlap import narration_units, rouge_l_f1, rouge_l_screen_coverage
from mmpquant.scoring.redundancy import RedundancyScorer

from .conftest import JaccardEmbedder, make_window


def stream(*screens, step=2.0):
    """screens: lists of texts shown together, one list per 2 s screen."""
    spans = [(i * step, (i + 1) * step) for i in range(len(screens))]
    return caption_stream([list(s) for s in screens], spans)


class TestCaptionStream:
    def test_a_caption_held_across_screens_is_one_piece(self):
        pieces = stream(["Every day, we face small"], ["Every day, we face small"])
        assert len(pieces) == 1 and pieces[0].end == 4.0

    def test_a_progressive_render_keeps_the_longest_reading(self):
        pieces = stream(["you unders"], ["you understand s"], ["you understand safety"])
        assert [p.text for p in pieces] == ["you understand safety"]

    def test_all_caps_titles_and_banners_never_join(self):
        pieces = stream(["SAFETY TIPS"], ["SUBSCRIBE"], ["Water boils"])
        assert [p.text for p in pieces] == ["Water boils"]


class TestStitching:
    def test_a_sentence_split_over_screens_is_rebuilt(self):
        pieces = stream(["Another important section of", "the course focuses"],
                        ["on fire prevention and", "emergency preparation."])
        out = stitch_sentences(pieces)
        assert [s.text for s in out] == [(
            "Another important section of the course focuses on fire prevention "
            "and emergency preparation.")]
        assert (out[0].start, out[0].end) == (0.0, 4.0)

    def test_a_one_screen_sentence_is_left_to_the_per_screen_classifier(self):
        assert stitch_sentences(stream(["Water boils at 100 degrees."])) == []

    def test_no_terminal_punctuation_no_sentence(self):
        assert stitch_sentences(stream(["Gaussian Elimination"], ["Signed Area = det(A)"])) == []

    def test_pieces_too_far_apart_in_time_do_not_join(self):
        pieces = caption_stream([["Heat flows from"], ["hot bodies to cold ones."]],
                                [(0.0, 2.0), (5.0, 7.0)])
        assert stitch_sentences(pieces, max_gap_s=1.0) == []

    def test_a_joined_fragment_that_is_not_a_sentence_is_dropped(self):
        assert stitch_sentences(stream(["Chapter three and"], ["more examples."])) == []

    def test_at_most_max_pieces_screens(self):
        pieces = stream(["Water"], ["that is heated"], ["in a sealed"], ["container"],
                        ["eventually boils."])
        out = stitch_sentences(pieces, max_pieces=4)
        assert all(len(s.pieces) <= 4 for s in out)


class TestWordMatchIsScreenCoverage:
    def test_a_screen_read_aloud_inside_longer_speech_scores_one(self):
        screen = "small safety risks often unnoticed"
        speech = "every day face small safety risks often unnoticed home work commuting"
        assert rouge_l_screen_coverage(screen, speech) == 1.0
        assert rouge_l_f1(screen, speech) < 0.7  # the old measure's ceiling

    def test_half_the_screen_spoken_is_half(self):
        assert rouge_l_screen_coverage("alpha beta gamma delta", "alpha beta") == 0.5


class TestMeaningIsSentenceToSentence:
    def test_narration_units_are_sentences_and_consecutive_pairs(self):
        assert narration_units("A b. C d. E f.") == ["A b.", "C d.", "E f.", "A b. C d.", "C d. E f."]
        assert narration_units("no punctuation here") == ["no punctuation here"]

    def test_surrounding_speech_does_not_dilute_the_match(self):
        sentence = "Heat cannot spontaneously flow from a colder body."
        narration = (f"Today we look at something else entirely and unrelated. {sentence} "
                     "Then we move on to many other topics of discussion.")
        w = make_window(sentence, narration)
        best = RedundancyScorer(embedder=JaccardEmbedder()).score_window(w)
        whole = RedundancyScorer(embedder=JaccardEmbedder(),
                                 semantic_measure="whole_window").score_window(w)
        assert best.components["semantic_raw"] == pytest.approx(1.0)
        assert whole.components["semantic_raw"] < 0.5

    def test_old_f1_is_still_recorded_for_comparison(self):
        w = make_window("Heat cannot spontaneously flow from a colder body.",
                        "So heat cannot spontaneously flow from a colder body, as you know.")
        ws = RedundancyScorer(embedder=JaccardEmbedder()).score_window(w)
        assert "verbatim_f1_raw" in ws.components
        assert ws.components["verbatim_raw"] >= ws.components["verbatim_f1_raw"]

    def test_unknown_measures_are_rejected(self):
        with pytest.raises(ValueError):
            RedundancyScorer(embedder=JaccardEmbedder(), verbatim_measure="bleu")


class TestWindowsGetStitchedSentences:
    def test_window_carries_the_stitched_sentence_and_drops_its_parts(self):
        from mmpquant.extract.align import OnscreenParams, WindowingParams, build_windows
        from mmpquant.types import AsrSegment, AsrWord, OcrBlock, OcrLine, Transcript

        def block(t0, t1, *texts):
            lines = tuple(OcrLine(t, 0.99, (100, 100 + 40 * i, 900, 130 + 40 * i))
                          for i, t in enumerate(texts))
            return OcrBlock(t0, t1, lines, t0, 1)

        blocks = (block(0, 3, "Every day, we face small", "safety risks that often go"),
                  block(3, 6, "unnoticed at home, at work,", "or even while commuting."))
        sentence = ("Every day we face small safety risks that often go unnoticed at "
                    "home at work or even while commuting")
        speech = sentence.split(" ")
        words = tuple(AsrWord(i * 0.4, i * 0.4 + 0.35, w) for i, w in enumerate(speech))
        tr = Transcript((AsrSegment(0, 10, " ".join(speech), words),), "en", 0.99, 10.0)
        [w] = build_windows(tr, blocks, 10.0, WindowingParams(size_s=10.0, tolerance_s=3.0,
                                                              expand="asr",
                                                              keep_partial_final_window=True),
                            onscreen=OnscreenParams())
        sentences = [e for e in w.ocr_elements if e["kind"] == "sentence"]
        assert [e["reason"] for e in sentences] == ["stitched_across_screens"]
        assert sentences[0]["text"].startswith("Every day, we face small safety risks")
        assert sentences[0]["text"].endswith("or even while commuting.")


class TestOnlyCaptionLikeTextJoins:
    """Guards added after the first corpus run glued chalkboards, bulleted
    slides and watermarks into "sentences" (15 Oct 2026)."""

    @pytest.mark.parametrize("text", [
        "and dossit qubit is do augthig if",               # misread chalk
        "Governing laws/theories •Thermodynamics",         # bulleted slide
        "clideo.com Subspace S is orthogonal",             # watermark
        "If f(0) = f(1) → constant",                       # notation
        "355855617715 learning modules that help",         # digit noise
        "SAFETY TIPS FOR WORK",                            # all-caps title
    ])
    def test_not_caption_like(self, text):
        from mmpquant.extract.captions import is_caption_like

        assert not is_caption_like(text)

    def test_typed_caption_text_is_caption_like(self):
        from mmpquant.extract.captions import is_caption_like

        assert is_caption_like("risks can quickly turn into serious emergencies.")

    def test_pieces_in_different_places_on_screen_do_not_join(self):
        top, bottom = (100, 50, 900, 120), (100, 600, 900, 670)
        pieces = caption_stream([[("Heat always flows from", top)],
                                 [("hot bodies to cold ones.", bottom)]],
                                [(0.0, 2.0), (2.0, 4.0)])
        assert stitch_sentences(pieces) == []

    def test_text_reappearing_later_is_a_new_piece_not_a_stretched_old_one(self):
        pieces = caption_stream([["Heat always flows"], ["Something else entirely"],
                                 ["Heat always flows"]],
                                [(0.0, 2.0), (2.0, 60.0), (60.0, 62.0)], max_gap_s=1.0)
        assert [(p.start, p.end) for p in pieces if p.text == "Heat always flows"] == [
            (0.0, 2.0), (60.0, 62.0)]
