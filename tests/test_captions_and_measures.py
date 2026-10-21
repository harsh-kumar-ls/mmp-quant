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
        # per-sentence values live in the sentence checks since 21 Oct 2026
        assert best.evidence["sentence_checks"][0]["semantic_raw"] == pytest.approx(1.0)
        assert whole.evidence["sentence_checks"][0]["semantic_raw"] < 0.5

    def test_old_f1_is_still_recorded_for_comparison(self):
        w = make_window("Heat cannot spontaneously flow from a colder body.",
                        "So heat cannot spontaneously flow from a colder body, as you know.")
        ws = RedundancyScorer(embedder=JaccardEmbedder()).score_window(w)
        assert "verbatim_f1_raw" in ws.components
        chk = ws.evidence["sentence_checks"][0]
        assert chk["verbatim_raw"] >= ws.components["verbatim_f1_raw"]

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


class TestHighlightedKeyLine:
    """User's ruling, 20 Oct 2026: one short sentence set apart in large type
    is a highlighted point (signaling), not redundancy. Whiteboard bubble
    phrases and short slide statements are not."""

    def _el(self, text, *, height=100, lines=1, reason="finite_verb"):
        from mmpquant.onscreen import KIND_SENTENCE, TextElement
        from mmpquant.types import OcrLine

        per = height / lines
        ls = tuple(OcrLine(f"{text} {i}", 0.99, (0, i * per, 800, (i + 1) * per))
                   for i in range(lines)) if lines > 1 else (OcrLine(text, 0.99, (0, 0, 800, height)),)
        return TextElement(lines=ls, kind=KIND_SENTENCE, bbox=(0, 0, 800, height),
                           evidence={"reason": reason})

    def _mark(self, *elements, usual=50):
        from mmpquant.extract.align import OnscreenParams, _mark_highlighted

        return _mark_highlighted(list(elements), usual, OnscreenParams())

    def test_one_short_line_in_large_type_is_a_label(self):
        [e] = self._mark(self._el("All areas get scaled by det(A)", height=100))
        assert e.kind == "label" and e.evidence["reason"] == "highlighted_key_line"
        assert e.evidence["size_ratio"] == 2.0

    def test_normal_size_text_stays_a_sentence(self):
        [e] = self._mark(self._el("All areas get scaled by det(A)", height=55))
        assert e.kind == "sentence"

    def test_several_lines_stay_a_sentence(self):
        # whiteboard bubble phrases are drawn over several lines
        [e] = self._mark(self._el("SCHOOLS ARE ORGANISED ON FACTORY LINES", height=200, lines=4))
        assert e.kind == "sentence"

    def test_a_long_line_stays_a_sentence(self):
        long = "This is a long sentence of well over twelve words that keeps on going here"
        [e] = self._mark(self._el(long, height=100))
        assert e.kind == "sentence"

    def test_stitched_captions_are_never_highlights(self):
        [e] = self._mark(self._el("Every day we face small risks.", height=100,
                                  reason="stitched_across_screens"))
        assert e.kind == "sentence"


class TestOnlyTheVisiblePartIsCompared:
    """21 Oct 2026 (expectation R10b): a stitched sentence is classified whole,
    but each window compares only the part of it on screen in that window."""

    def test_visible_text_is_the_pieces_on_screen_then(self):
        pieces = stream(["But when people are not"], ["prepared, these everyday"],
                        ["risks can quickly turn into serious emergencies."])
        [s] = stitch_sentences(pieces)
        assert s.visible_text(0.0, 4.0) == "But when people are not prepared, these everyday"
        assert s.visible_text(4.0, 6.0) == "risks can quickly turn into serious emergencies."
        assert s.visible_text(0.0, 6.0) == s.text

    def test_scorer_does_not_reclassify_a_visible_fragment(self):
        from mmpquant.scoring.redundancy import RedundancyScorer
        from mmpquant.types import Window

        w = Window(index=0, start=0, end=10, asr_text="but when people are not prepared these everyday",
                   ocr_text="", asr_pool_start=-3, asr_pool_end=13,
                   ocr_sentence_text="But when people are not prepared, these everyday",
                   ocr_elements=({"text": "But when people are not prepared, these everyday",
                                  "kind": "sentence", "reason": "stitched_across_screens"},))
        ws = RedundancyScorer(embedder=JaccardEmbedder()).score_window(w)
        assert ws.evidence["gate"] == "passed"
        assert ws.evidence["sentence_checks"][0]["verbatim_raw"] == pytest.approx(1.0)


class TestMatchingFixes21Oct:
    """Three measurement faults found through expectation R10b (Test_28)."""

    def test_a_word_counts_as_said_if_lemma_or_surface_matches(self):
        from mmpquant.onscreen import said_forms, shared_content

        said = said_forms("But when people are not prepared, these everyday risks grow.")
        words, shared = shared_content("But when people are not prepared, these everyday", said)
        assert words == shared  # "prepared" matches whatever the tagger made of it

    def test_word_match_does_not_depend_on_the_order_sentences_sit_on_screen(self):
        from mmpquant.scoring.redundancy import RedundancyScorer

        a = "The challenge is that traditional safety training can feel overwhelming."
        b = "Risks can quickly turn into serious emergencies."
        narration = f"{b} {a}"   # spoken in the other order
        sc = RedundancyScorer(embedder=JaccardEmbedder())
        assert sc._verbatim_sentences([a, b], "", narration) == pytest.approx(1.0)

    def test_percent_sign_reads_as_percent(self):
        from mmpquant.textnorm import normalise_for_verbatim

        assert normalise_for_verbatim("by up to 17%") == normalise_for_verbatim("by up to 17 percent")


def test_a_numbered_list_is_not_caption_text():
    from mmpquant.extract.captions import is_caption_like

    assert not is_caption_like("PEDAGOGY 1. Applied Educational Theory 2. Classroom Management")
    assert is_caption_like("Choosing the right assessment method is part of pedagogy.")


class TestSplitCompounds:
    """21 Oct 2026, user: Test_28's narrator says 'microlearning'; the
    transcript wrote 'micro -learning' — a transcription fault."""

    def test_gate_counts_a_compound_the_recogniser_split(self):
        from mmpquant.onscreen import said_forms, shared_content

        said = said_forms("The micro -learning safety everywhere course")
        _words, shared = shared_content("The Microlearning Safety Everywhere course", said)
        assert "microlearning" in shared

    def test_word_match_joins_the_split_halves_only_when_the_screen_has_the_compound(self):
        from mmpquant.overlap import rouge_l_screen_coverage

        assert rouge_l_screen_coverage("microlearning safety", "micro -learning safety") == 1.0
        # "real - life" on screen stays two words; the transcript's "real -life" must still match
        assert rouge_l_screen_coverage("real - life safety", "real -life safety") == 1.0


class TestTimeWeightedWindowScore:
    """21 Oct 2026 (user; expectation R12): each on-screen sentence counts for
    the seconds it is visible, unrestated sentences count 0, seconds with no
    sentence on screen count 0."""

    def _window(self, elements, asr):
        from mmpquant.types import Window

        return Window(index=1, start=10.0, end=20.0, asr_text=asr, ocr_text="",
                      asr_pool_start=7, asr_pool_end=23,
                      ocr_sentence_text=" ".join(e["text"] for e in elements),
                      ocr_elements=tuple(elements))

    def _scorer(self):
        from mmpquant.scoring.redundancy import RedundancyScorer
        return RedundancyScorer(embedder=JaccardEmbedder())

    def test_a_caption_on_screen_briefly_counts_briefly(self):
        big = "The big wheel you can see is the Millennium Wheel or London Eye."
        high = "It is 135 metres high with 32 glass capsules."
        els = [{"text": big, "kind": "sentence", "spans": [[7.0, 11.5]]},
               {"text": high, "kind": "sentence", "spans": [[11.5, 17.0]]}]
        ws = self._scorer().score_window(self._window(els, f"{big} {high}"))
        vis = {c["sentence"][:7]: c["visible_s"] for c in ws.evidence["sentence_checks"]}
        assert vis == {"The big": 1.5, "It is 1": 5.5}
        # both read aloud; 7 of 10 s have a sentence on screen (recorded), and
        # the score averages over those 7 s only
        assert ws.components["screen_time_share"] == pytest.approx(0.7)
        assert ws.score > 85

    def test_an_unread_sentence_pulls_the_window_down(self):
        said = "Heat cannot spontaneously flow from a colder body to a hotter body."
        unread = "Entropy of an isolated system never decreases over time at all."
        els = [{"text": said, "kind": "sentence", "spans": [[10.0, 20.0]]},
               {"text": unread, "kind": "sentence", "spans": [[10.0, 20.0]]}]
        both = self._scorer().score_window(self._window(els, said))
        alone = self._scorer().score_window(self._window(els[:1], said))
        assert both.score < alone.score

    def test_seconds_with_no_sentence_on_screen_are_left_out_of_the_average(self):
        said = "Heat cannot spontaneously flow from a colder body to a hotter body."
        full = [{"text": said, "kind": "sentence", "spans": [[10.0, 20.0]]}]
        half = [{"text": said, "kind": "sentence", "spans": [[15.0, 20.0]]}]
        a = self._scorer().score_window(self._window(full, said))
        b = self._scorer().score_window(self._window(half, said))
        assert b.score == pytest.approx(a.score, abs=0.5)
        assert b.components["screen_time_share"] == pytest.approx(0.5)


@pytest.mark.parametrize("sentence, question", [
    ("When it opened on 10th June, the Bridge had to be closed after just a few days", False),
    ("How we learn depends on what we already know.", False),
    ("What is the new volume of the solution?", True),
    ("What is the new volume of the solution", True),       # OCR lost the "?"
    ("How do we square that circle?", True),
    ("Why does ice float on water", True),
    ("(b) What is the new volume of the solution", True),   # worksheet item label
])
def test_a_statement_opening_with_a_question_word_is_not_a_question(sentence, question):
    from mmpquant.onscreen import is_question

    assert is_question(sentence) is question


class TestTimingFixes21OctLondon:
    """21 Oct 2026, animation_london_subtitled (user: "catching the narration
    timestamp and the on-screen text with timestamp is the most crucial part")."""

    def test_narration_follows_the_whole_showing_across_the_window_edge(self):
        # a subtitle on screen 106-114 s, spoken from 105.9 s; the 110-120 s
        # window sees it only from 110 s, yet must hear it from the start
        from mmpquant.scoring.base import VideoContext
        from mmpquant.types import AsrSegment, AsrWord, Transcript, Window

        text = "It is a tourist attraction and you can walk the 530 steps up."
        words = tuple(AsrWord(105.9 + 0.5 * i, 106.3 + 0.5 * i, w)
                      for i, w in enumerate(text.split()))
        tr = Transcript((AsrSegment(105.9, 113.0, text, words),), "en", 0.99, 130.0)

        def window(i, lo, spans):
            return Window(index=i, start=lo, end=lo + 10, asr_text="", ocr_text=text,
                          ocr_sentence_text=text, asr_pool_start=lo - 3, asr_pool_end=lo + 13,
                          ocr_elements=({"text": text, "kind": "sentence", "spans": spans},))

        wins = [window(10, 100.0, [[106.0, 110.0]]), window(11, 110.0, [[110.0, 114.0]])]
        scorer = RedundancyScorer(embedder=JaccardEmbedder())
        scorer.score(wins, context=VideoContext(tr, None))
        heard, lo, hi = scorer._narration_for(text, {text: [[110.0, 114.0]]}, wins[1])
        assert heard.startswith("It is a tourist")
        assert (lo, hi) == (103.0, 117.0)  # whole showing 106-114 s, +-3 s

    def test_a_separate_later_showing_is_not_joined(self):
        from mmpquant.types import Window

        text = "Nelson's column is a monument to Admiral Nelson."
        w = Window(index=0, start=0, end=10, asr_text="", ocr_text=text, ocr_sentence_text=text,
                   asr_pool_start=-3, asr_pool_end=13,
                   ocr_elements=({"text": text, "kind": "sentence", "spans": [[2.0, 4.0]]},))
        far = Window(index=5, start=50, end=60, asr_text="", ocr_text=text,
                     ocr_sentence_text=text, asr_pool_start=47, asr_pool_end=63,
                     ocr_elements=({"text": text, "kind": "sentence",
                                    "spans": [[55.0, 57.0]]},))
        showings = RedundancyScorer(embedder=JaccardEmbedder())._video_showings([w, far])
        assert list(showings.values()) == [[[2.0, 4.0], [55.0, 57.0]]]

    def test_a_slide_shown_all_video_is_not_compared_with_all_narration(self):
        from mmpquant.scoring.base import VideoContext
        from mmpquant.types import AsrSegment, AsrWord, Transcript, Window

        text = "Heat cannot spontaneously flow from a colder body to a hotter body."
        words = tuple(AsrWord(float(t), t + 0.4, f"w{t}") for t in range(0, 160, 2))
        tr = Transcript((AsrSegment(0, 160, " ".join(w.text for w in words), words),),
                        "en", 0.99, 160.0)
        wins = [Window(index=i, start=10.0 * i, end=10.0 * i + 10, asr_text="", ocr_text=text,
                       ocr_sentence_text=text, asr_pool_start=10 * i - 3,
                       asr_pool_end=10 * i + 13,
                       ocr_elements=({"text": text, "kind": "sentence",
                                      "spans": [[10.0 * i, 10.0 * i + 10]]},))
                for i in range(16)]
        scorer = RedundancyScorer(embedder=JaccardEmbedder())
        scorer.score(wins, context=VideoContext(tr, None))
        _, lo, hi = scorer._narration_for(text, {text: [[50.0, 60.0]]}, wins[5])
        # nearby rule (user, 22 Oct 2026): one window either side, +-3 s --
        # not the whole video's narration
        assert (lo, hi) == (37.0, 73.0)

    def test_a_slide_read_once_stops_counting_one_window_later(self):
        # a static slide is ONE OCR block for its whole showing (Test_16:
        # 0:00-0:42); read at 0:01-0:06 it counts in 0:00-0:10 and, by the
        # nearby rule, 0:10-0:20 -- not at 0:20-0:30
        from mmpquant.scoring.base import VideoContext
        from mmpquant.types import AsrSegment, AsrWord, Transcript, Window

        text = "A 0.8 M HCl solution was diluted to a final volume of 500 mL."
        said = text.split()
        words = tuple(AsrWord(1.0 + 0.4 * i, 1.3 + 0.4 * i, w) for i, w in enumerate(said))
        tr = Transcript((AsrSegment(1, 6, text, words),), "en", 0.99, 60.0)
        wins = [Window(index=i, start=10.0 * i, end=10.0 * i + 10,
                       asr_text=text if i == 0 else "now we work through the steps",
                       ocr_text=text, ocr_sentence_text=text, asr_pool_start=10 * i - 3,
                       asr_pool_end=10 * i + 13,
                       ocr_elements=({"text": text, "kind": "sentence",
                                      "spans": [[0.0, 42.5]]},))
                for i in range(5)]
        scorer = RedundancyScorer(embedder=JaccardEmbedder())
        result = scorer.score(wins, context=VideoContext(tr, None))
        check = result.window_scores[2].evidence["sentence_checks"][0]
        assert (check["narration_from"], check["narration_to"]) == (7.0, 43.0)
        assert result.window_scores[2].score == 0
        assert result.window_scores[0].score > 50
        assert result.window_scores[1].score > 50

    def test_the_reading_on_screen_longest_wins_among_near_copies(self):
        from mmpquant.onscreen import KIND_SENTENCE, TextElement, collapse_near_duplicates
        from mmpquant.types import OcrLine

        def el(text, spans):
            return TextElement(lines=(OcrLine(text, 1.0, None),), kind=KIND_SENTENCE,
                               bbox=None, evidence={"spans": spans})

        usual = "It is a tourist attraction and you can walk the 530 steps"
        stray = "It is a tourist attraction and you can walk. the 530 steps"
        out = collapse_near_duplicates([el(usual, [[106.0, 109.5]]), el(stray, [[109.5, 110.0]])])
        assert [e.text for e in out] == [usual]
        assert out[0].evidence["spans"] == [[106.0, 110.0]]

    def test_a_lone_glyph_does_not_chain_a_watermark_to_a_subtitle(self):
        from mmpquant.onscreen import group_lines
        from mmpquant.types import OcrLine

        lines = [OcrLine("escher Editor", 1.0, (556, 362, 802, 401)),
                 OcrLine("L", 1.0, (794, 420, 830, 480)),
                 OcrLine("who died at the Battle", 1.0, (418, 544, 862, 596)),
                 OcrLine("of Trafalgar in 1815.", 1.0, (436, 602, 848, 658))]
        assert [[ln.text for ln in g] for g in group_lines(lines)] == [
            ["escher Editor"], ["L"], ["who died at the Battle", "of Trafalgar in 1815."]]

    def test_one_oversized_box_does_not_bridge_a_wide_gap(self):
        from mmpquant.onscreen import group_lines
        from mmpquant.types import OcrLine

        lines = [OcrLine("© Loescher Editore", 1.0, (462, 362, 816, 400)),
                 OcrLine("You can also walk along the top bridge", 1.0, (249, 529, 1031, 625))]
        assert len(group_lines(lines)) == 2
        # ordinary wrapped lines still join
        wrapped = [OcrLine("You can also walk along the top bridge", 1.0, (250, 552, 1026, 602)),
                   OcrLine("that joins the two towers.", 1.0, (388, 606, 894, 658))]
        assert len(group_lines(wrapped)) == 1


def test_the_window_score_is_the_time_weighted_sum_of_sentence_scores():
    """The report shows window = sum(time_weight x sentence score); it must add up."""
    from mmpquant.types import Window

    big = "The big wheel you can see is the Millennium Wheel or London Eye."
    high = "It is 135 metres high with 32 glass capsules."
    unread = "Entropy of an isolated system never decreases over time at all."
    els = ({"text": big, "kind": "sentence", "spans": [[7.0, 11.5]]},
           {"text": high, "kind": "sentence", "spans": [[11.5, 17.0]]},
           {"text": unread, "kind": "sentence", "spans": [[12.0, 20.0]]})
    w = Window(index=1, start=10.0, end=20.0, asr_text=f"{big} {high}", ocr_text="",
               asr_pool_start=7, asr_pool_end=23, ocr_elements=els,
               ocr_sentence_text=" ".join(e["text"] for e in els))
    ws = RedundancyScorer(embedder=JaccardEmbedder()).score_window(w)
    checks = ws.evidence["sentence_checks"]
    assert sum(c["time_weight"] for c in checks) == pytest.approx(1.0, abs=1e-3)
    assert sum(c["time_weight"] * c["score"] for c in checks) == pytest.approx(ws.score, abs=0.05)
