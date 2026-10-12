"""Text on screen is not redundancy by being there (04 Oct 2026).

The user's finding on Test_11: a Seinfeld joke on screen and the unrelated
narration "In a previous video I've talked" scored 11.9. Meaning-similarity
never reaches 0 for two English passages (unrelated sentences score 0.1-0.3),
and ROUGE-L counted shared function words ("is", "it", "so", "your").

The fix, tested here: a window scores 0 unless the narration actually says the
on-screen content words -- at least two more than narration elsewhere in the
video does -- and above that gate both measures are corrected for chance.
Strings below are real corpus text unless noted.
"""

from __future__ import annotations

import pytest

from mmpquant.scoring.redundancy import RedundancyScorer, _above_chance
from mmpquant.textnorm import normalise_for_verbatim

from .conftest import ConstantEmbedder, make_window

SEINFELD = (
    "Jerry: Ah, you're crazy!\nKramer: Am I? Or am I so sane that\nyou just blew your mind?\n"
    "Jerry: It's impossible!\nKramer: Is it?! Or is it so possible\nyour head is spinning like a top?"
)
PROBLEM = (
    "The concentration of a 300mL aqueous KCl solution was increased from 0.25M to 0.75M "
    "by evaporation. What is the new volume of the solution?"
)


def scorer(similarity=0.9):
    return RedundancyScorer(embedder=ConstantEmbedder(similarity))


class TestTheGate:
    def test_test_11_seinfeld_window_scores_zero(self):
        # Even with the embedder claiming strong similarity: no shared content,
        # no redundancy.
        ws = scorer(0.9).score_window(make_window(SEINFELD, "In a previous video I've talked"))
        assert ws.score == 0.0
        assert ws.evidence["gate"] == "failed"
        assert ws.evidence["shared_content_words"] == []

    def test_function_words_alone_do_not_count(self):
        ws = scorer().score_window(
            make_window("It is what it is, and so it goes", "so it is that it was")
        )
        assert ws.score == 0.0

    def test_text_read_aloud_passes(self):
        narration = "The concentration of a 300 mL aqueous KCl solution was increased by evaporation"
        ws = scorer().score_window(make_window(PROBLEM, narration))
        assert ws.evidence["gate"] == "passed"
        assert ws.score > 0

    def test_shared_words_are_recorded_so_a_reader_can_check_them(self):
        ws = scorer().score_window(make_window(
            PROBLEM, "the concentration of a 300 mL aqueous KCl solution was increased "
                     "from 0 .25 to 0 .75 by evaporation"))
        assert {"concentration", "increase", "0.25"} <= set(ws.evidence["shared_content_words"])
        assert ws.evidence["restated_sentences"][0]["sentence"].startswith("The concentration")

    def test_a_third_of_a_sentence_is_not_a_restatement(self):
        # 3 of the sentence's 9 content words. Sentence-level redundancy means
        # the narration says most of a sentence, not a few of its words.
        ws = scorer().score_window(make_window(PROBLEM, "the concentration was increased from 0 .25"))
        assert ws.score == 0.0

    def test_a_gated_window_still_counts_as_assessed(self):
        # Scored 0, not excluded: it was assessed and found not redundant,
        # which pulls its segment's average down as it should.
        ws = scorer().score_window(make_window(SEINFELD, "In a previous video I've talked"))
        assert ws.qualifies is True
        assert ws.exclusion_reason is None


class TestChanceCorrection:
    def windows(self, concurrent):
        # The problem is on screen throughout; the narration solves it,
        # mentioning concentration and volume everywhere. Only the window
        # under test reads the problem aloud.
        elsewhere = "so the concentration goes up and the volume must go down"
        ws = [make_window(PROBLEM, elsewhere, index=i, start=i * 10.0, end=i * 10.0 + 10)
              for i in range(8)]
        ws[0] = make_window(PROBLEM, concurrent, index=0, start=0.0, end=10.0)
        return ws

    def test_talking_about_the_topic_is_not_reading_it(self):
        # Shares "concentration" and "volume" -- exactly what the rest of the
        # video shares. Nothing beyond chance.
        result = scorer().score(self.windows("the concentration and the volume are related"))
        assert result.window_scores[0].score == 0.0

    def test_reading_it_aloud_beats_chance(self):
        result = scorer().score(self.windows(
            "the concentration of a 300 mL aqueous KCl solution was increased from 0.25 to 0.75 "
            "by evaporation what is the new volume"))
        assert result.window_scores[0].evidence["gate"] == "passed"
        assert result.window_scores[0].score > 0

    def test_too_few_other_windows_means_no_baseline(self):
        ws = scorer().score([make_window(PROBLEM, "concentration volume solution evaporation")])
        assert ws.window_scores[0].evidence["null_windows"] == 0

    @pytest.mark.parametrize("value,chance,expected", [
        (0.8, 0.6, 0.5), (0.6, 0.6, 0.0), (0.4, 0.6, 0.0), (0.5, 0.0, 0.5), (0.9, 1.0, 0.0),
    ])
    def test_above_chance_arithmetic(self, value, chance, expected):
        assert _above_chance(value, chance) == pytest.approx(expected)


class TestNumbersMatch:
    """Read-aloud content in a worked problem is mostly numbers, and none of
    them matched: decimals were split, units glued on, and Whisper writes
    "0 .25"."""

    @pytest.mark.parametrize("ocr,asr", [
        ("increased from 0.25M to 0.75M", "increased from 0 .25 to 0 .75"),
        ("300mL", "300"),
    ])
    def test_ocr_and_asr_numbers_normalise_alike(self, ocr, asr):
        o, a = normalise_for_verbatim(ocr).split(), normalise_for_verbatim(asr).split()
        assert set(a) <= set(o)

    def test_decimal_point_survives_punctuation_stripping(self):
        assert normalise_for_verbatim("Version 2.0 of item 3.") == "version 2.0 of item 3"


class TestTopicVocabularySentenceReadAloud:
    def test_14jz_strang_reading_what_he_writes_passes(self):
        # Every content word of this sentence is the lecture's own vocabulary,
        # so narration elsewhere shares some of them by chance. Reading it
        # aloud shares more -- that must be enough. The first version of the
        # gate demanded two words *on top of* chance and scored this 0.
        screen = "Subspace S is orthogonal to Subspace T means every vector in S is orthogonal"
        elsewhere = "and orthogonal things come up again here"
        ws = [make_window(screen, elsewhere, index=i, start=i * 10.0, end=i * 10.0 + 10)
              for i in range(8)]
        ws[0] = make_window(screen, "It means that every vector in S is orthogonal to every vector in T")
        result = RedundancyScorer(embedder=ConstantEmbedder(0.9)).score(ws)
        assert result.window_scores[0].evidence["gate"] == "passed"
        assert result.window_scores[0].score > 0



class TestSentenceLevel:
    """05 Oct 2026, test4: words gathered from all over a screen are not a
    restated sentence."""

    PAPER = (
        "J. Sleep Res. (1992) 1, 231-239\n"
        "24-h variation of vigilance in the cockroach Blaberus giganteus\n"
        "IRENE TOBLER and MARTIN NEUNER-JEHLE\n"
        "Institute of Pharmacology, University of Zurich, Switzerland\n"
        "Our objective was to investigate whether sleep-like states occur in the cockroach, "
        "Blaberus giganteus. The cockroaches showed a nocturnal behaviour exhibiting a bout of "
        "locomotion at dark onset which lasted several hours."
    )

    def test_test4_scattered_words_score_zero(self):
        narration = ("cracks in this brain-centric view started to show when the Swiss scientist "
                     "Irene Tobler noticed that cockroaches sleep.")
        ws = scorer(0.9).score_window(make_window(self.PAPER, narration))
        assert ws.score == 0.0
        assert ws.evidence["restated_sentences"] == []

    def test_reading_one_bullet_of_a_slide_counts(self):
        # Test_18: a three-bullet slide; the narrator reads one bullet. Bullets
        # are separate sentences, so one read aloud is a full restatement.
        slide = ("\u2022 Declarative knowledge is statements of fact\n"
                 "\u2022 Imperative knowledge is a recipe or how-to\n"
                 "\u2022 Programming is about writing recipes")
        ws = scorer().score_window(make_window(slide, "Um, declarative knowledge is a statement of fact"))
        assert ws.score > 0
        assert ws.evidence["restated_sentences"][0]["sentence"] == (
            "Declarative knowledge is statements of fact")

    def test_author_lines_and_headings_are_not_sentences(self):
        ws = scorer().score_window(make_window(
            "IRENE TOBLER and MARTIN NEUNER-JEHLE\nJ. Sleep Res. (1992) 1, 231-239",
            "Irene Tobler and Martin Neuner-Jehle wrote in the Journal of Sleep Research"))
        assert ws.score == 0.0


class TestAnsweringAQuestionIsNotRestatingIt:
    """User's ruling, 05 Oct 2026: a question on screen and the narrator's
    answer to it are separate things; the answer is not redundancy, even when
    it repeats the question's words. Real lines from Test_17."""

    QUESTION = "(a) What is the new volume of the solution?"

    def test_the_answer_does_not_count(self):
        ws = scorer().score_window(make_window(
            self.QUESTION, "So that's the new volume of the solution. It is 100 milliliters."))
        assert ws.score == 0.0

    def test_another_answer_does_not_count(self):
        ws = scorer().score_window(make_window(
            "(b) How much water was removed?",
            "The negative sign indicates that water was removed."))
        assert ws.score == 0.0

    def test_reading_the_question_aloud_still_counts(self):
        ws = scorer().score_window(make_window(
            "(b) How much water was removed?", "Now how much water was removed? So we went from 300 to 100."))
        assert ws.score > 0
        assert ws.evidence["restated_sentences"][0]["sentence"].endswith("removed?")

    def test_statements_are_unaffected(self):
        ws = scorer().score_window(make_window(
            "Declarative knowledge is statements of fact",
            "Declarative knowledge is a statement of fact."))
        assert ws.score > 0


class TestEveryCheckIsRecorded:
    def test_failed_checks_are_kept_for_the_breakdown(self):
        ws = scorer().score_window(make_window(
            "Declarative knowledge is statements of fact", "We talk about knowledge today."))
        checks = ws.evidence["sentence_checks"]
        assert len(checks) == 1
        assert checks[0]["passed"] is False
        assert checks[0]["shared_words"] == ["knowledge"]
