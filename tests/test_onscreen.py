"""On-screen text grouping and classification.

These tests encode the operative definition of redundancy as clarified by the
supervisor on 07 Sep 2026: redundancy is on-screen *prose* duplicating
concurrent narration, and keywords, topic titles and labels are not redundancy
however many of them are on screen.

The first class is a regression suite built from a real failure: an animated
video scored 8.1 when it should have been near zero, because three separate
diagram labels were summed into one 20-word "text block".
"""

from __future__ import annotations

from typing import ClassVar

from mmpquant.onscreen import (
    KIND_LABEL,
    KIND_SENTENCE,
    classify_text,
    collapse_near_duplicates,
    dedupe_lines,
    group_into_elements,
    sentence_text,
)
from mmpquant.types import OcrLine


def line(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrLine:
    return OcrLine(text=text, confidence=0.95, bbox=(x0, y0, x1, y1))


class TestScatteredLabelsAreNotProse:
    """Regression: the Kurzgesagt failure, in miniature."""

    def test_three_separate_labels_stay_three_labels(self):
        # Positions taken from the real frame: three signs at different heights
        # across the width of an animated scene.
        lines = [
            line("SPONSORS", 160, 430, 400, 480),
            line("& GRANTS", 160, 480, 400, 520),
            line("COMMERCIAL", 455, 370, 745, 415),
            line("WORK", 455, 415, 745, 455),
            line("FUNK", 775, 355, 900, 400),
        ]
        elements = group_into_elements(lines, min_prose_words=5)
        # Three spatially distinct signs -> three elements, not one.
        assert len(elements) == 3
        # "& GRANTS" is one word once punctuation is stripped
        assert [e.word_count for e in elements] == [2, 2, 1]
        assert all(e.kind == KIND_LABEL for e in elements)

    def test_summing_labels_would_have_qualified_them(self):
        # Demonstrates the defect being guarded against: the same five lines
        # total 6 words, which clears a 5-word threshold when summed.
        from mmpquant.textnorm import word_count

        lines = [
            line("SPONSORS", 160, 430, 400, 480),
            line("& GRANTS", 160, 480, 400, 520),
            line("COMMERCIAL", 455, 370, 745, 415),
            line("WORK", 455, 415, 745, 455),
            line("FUNK", 775, 355, 900, 400),
        ]
        summed = word_count("\n".join(ln.text for ln in lines))
        assert summed >= 5, "the old rule would have called this prose"
        assert sentence_text(group_into_elements(lines, min_prose_words=5)) == ""

    def test_label_only_frame_contributes_no_compared_text(self):
        # Labels are excluded from the compared text, not merely from the
        # qualification decision, so they cannot raise the overlap score.
        lines = [line("ANIMATION", 100, 100, 300, 140)]
        assert sentence_text(group_into_elements(lines, min_prose_words=5)) == ""


class TestGrouping:
    def test_wrapped_sentence_is_one_element(self):
        # Two left-aligned lines, tight vertical gap: one sentence that wrapped.
        lines = [
            line("Mitochondria generate ATP through", 100, 200, 700, 240),
            line("oxidative phosphorylation", 100, 244, 520, 284),
        ]
        elements = group_into_elements(lines, min_prose_words=5)
        assert len(elements) == 1
        assert elements[0].kind == KIND_SENTENCE
        assert elements[0].word_count == 6

    def test_bullet_block_is_one_element(self):
        # A stacked, left-aligned list groups as a single block regardless of
        # what it classifies as -- grouping and classification are separate
        # concerns, and this test is about the former.
        lines = [
            line("Reduce carbon emissions", 120, 300, 520, 336),
            line("Improve public transit", 120, 340, 500, 376),
            line("Expand green spaces", 120, 380, 490, 416),
        ]
        elements = group_into_elements(lines, min_prose_words=5)
        assert len(elements) == 1
        # Three imperative bullet headers, no finite verb between them (English
        # imperatives use the bare form) -- a topic list, correctly a label.
        assert elements[0].kind == KIND_LABEL

    def test_large_vertical_gap_splits_elements(self):
        # A title at the top and a caption at the bottom are not one block.
        lines = [
            line("Photosynthesis", 100, 60, 400, 110),
            line("Figure 3 shows the light reaction", 100, 600, 700, 640),
        ]
        assert len(group_into_elements(lines, min_prose_words=5)) == 2

    def test_side_by_side_columns_stay_separate(self):
        # Same vertical band, opposite halves of the frame: two captions.
        lines = [
            line("left hand caption text", 50, 300, 350, 340),
            line("right hand caption text", 800, 300, 1100, 340),
        ]
        assert len(group_into_elements(lines, min_prose_words=5)) == 2

    def test_centred_title_wrapping_is_one_element(self):
        # Not left-aligned, but horizontally overlapping and vertically tight.
        lines = [
            line("HOW TO MAKE A", 260, 235, 970, 300),
            line("KURZGESAGT VIDEO", 175, 370, 1055, 460),
        ]
        elements = group_into_elements(lines, min_prose_words=5)
        assert len(elements) == 1

    def test_missing_bounding_boxes_keep_lines_separate(self):
        # Without layout there is nothing to reason about. Splitting risks
        # under-reporting; fusing risks inventing prose. Splitting is chosen.
        lines = [
            OcrLine(text="alpha beta", confidence=0.9, bbox=None),
            OcrLine(text="gamma delta", confidence=0.9, bbox=None),
        ]
        assert len(group_into_elements(lines, min_prose_words=5)) == 2

    def test_empty_input(self):
        assert group_into_elements([], min_prose_words=5) == []

    def test_blank_lines_are_dropped(self):
        lines = [line("   ", 0, 0, 10, 10), line("real text here now", 0, 20, 300, 60)]
        elements = group_into_elements(lines, min_prose_words=4)
        assert len(elements) == 1


class TestClassification:
    def test_word_count_alone_never_promotes_verbless_text_to_a_sentence(self):
        # Defect 10.1 (30 Sep 2026): a verbless-but-long fallback used to sit
        # here, and it never once caught a genuine verbless statement when
        # measured against the corpus -- only title cards, OCR garbage and
        # equations, all of which are long by construction and none of which
        # are prose. Length no longer promotes anything; ten words with no
        # finite verb is a label exactly like one word with no finite verb.
        five = group_into_elements([line("one two three four five", 0, 0, 500, 40)],
                                   min_prose_words=5)
        ten = group_into_elements(
            [line("one two three four five six seven eight nine ten", 0, 0, 900, 40)],
            min_prose_words=5,
        )
        assert five[0].kind == KIND_LABEL
        assert ten[0].kind == KIND_LABEL

    def test_min_prose_words_no_longer_changes_the_verdict(self):
        # The parameter is kept (it still flows into the audit-trail word
        # count and the scorer's own signaling-exception check downstream),
        # but it no longer decides sentence-vs-label on its own -- only a
        # finite verb does.
        lines = [line("one two three", 0, 0, 300, 40)]
        assert group_into_elements(lines, min_prose_words=3)[0].kind == KIND_LABEL
        assert group_into_elements(lines, min_prose_words=100)[0].kind == KIND_LABEL

    def test_sentence_text_keeps_only_prose(self):
        lines = [
            line("ANIMATION", 100, 60, 300, 100),
            line("The light reaction splits water molecules", 100, 400, 800, 440),
        ]
        elements = group_into_elements(lines, min_prose_words=5)
        out = sentence_text(elements)
        assert "light reaction" in out
        assert "ANIMATION" not in out

    def test_element_serialises_for_the_audit_trail(self):
        # Uses a real sentence (finite verb) so this test is about the
        # to_dict() shape, not a re-check of the classification rule.
        elements = group_into_elements([line("Water boils rapidly", 0, 0, 500, 40)],
                                       min_prose_words=5)
        d = elements[0].to_dict()
        assert d["kind"] == KIND_SENTENCE
        assert d["word_count"] == 3
        assert d["bbox"] == [0, 0, 500, 40]


class TestNearDuplicateCollapse:
    def test_animation_partial_renders_collapse(self):
        # The real sequence from an animated title card, captured at 2 fps as it
        # rendered in. One six-word title, read four times.
        variants = ["HOWTOMAKEA", "HOWTOI MAKEA", "HOWTO MAKEA", "HOW TO MAKE A"]
        elements = group_into_elements(
            [line(v, 100, 100 + i * 500, 600, 140 + i * 500) for i, v in enumerate(variants)],
            min_prose_words=5,
        )
        assert len(elements) == 4, "grouping keeps them apart (far apart vertically)"
        collapsed = collapse_near_duplicates(elements)
        assert len(collapsed) == 1
        assert collapsed[0].text == "HOW TO MAKE A"  # the most complete reading

    def test_exact_repeats_collapse(self):
        elements = group_into_elements(
            [
                line("SPONSORS & GRANTS", 100, 100, 400, 140),
                line("SPONSORS & GRANTS", 100, 900, 400, 940),
            ],
            min_prose_words=5,
        )
        assert len(collapse_near_duplicates(elements)) == 1

    def test_distinct_text_is_preserved(self):
        elements = group_into_elements(
            [
                line("COMMERCIAL WORK", 100, 100, 400, 140),
                line("SPONSORS AND GRANTS", 100, 900, 400, 940),
            ],
            min_prose_words=5,
        )
        assert len(collapse_near_duplicates(elements)) == 2

    def test_similar_but_different_labels_are_not_collapsed(self):
        # Guard on the fuzzy test: these share a word but are different labels.
        elements = group_into_elements(
            [
                line("ILLUSTRATION", 100, 100, 400, 140),
                line("ANIMATION", 100, 900, 400, 940),
            ],
            min_prose_words=5,
        )
        assert len(collapse_near_duplicates(elements)) == 2

    def test_original_order_is_preserved(self):
        elements = group_into_elements(
            [
                line("first distinct thing", 100, 100, 400, 140),
                line("second distinct thing", 100, 900, 400, 940),
            ],
            min_prose_words=5,
        )
        collapsed = collapse_near_duplicates(elements)
        assert collapsed[0].text.startswith("first")

    def test_empty_collapses_to_empty(self):
        assert collapse_near_duplicates([]) == []


class TestFiniteVerbCriterion:
    """The decisive test is grammatical completeness, not length.

    Examples are taken from the operational criteria supplied by the supervisor
    and its cited sources (Mayer 2017; Yue, Bjork & Bjork 2013).
    """

    LABELS: ClassVar[list[str]] = [
        "Ohm's Law",
        "Voltage divider",
        "Photosynthesis stages",
        "Step 3: Filtering",
        "Brake fluid pressure",
    ]
    SENTENCES: ClassVar[list[str]] = [
        "In a voltage divider, the output voltage is proportional to the resistance ratio",
        "Mitochondria generate ATP through oxidative phosphorylation",
        "The electron transport chain pumps protons across the inner membrane",
    ]

    def test_literature_examples_of_key_terms_are_labels(self):
        for text in self.LABELS:
            kind, _ = classify_text(text)
            assert kind == KIND_LABEL, f"{text!r} should be signaling, not redundancy"

    def test_literature_examples_of_sentences_are_sentences(self):
        for text in self.SENTENCES:
            kind, evidence = classify_text(text)
            assert kind == KIND_SENTENCE, f"{text!r} should be redundancy-eligible"
            assert evidence["finite_verbs"], f"no finite verb found in {text!r}"

    def test_headline_longer_than_the_threshold_is_still_a_label(self):
        # THE case length alone gets wrong. Six words, but "make" is an
        # infinitive: the phrase names a topic, it does not assert anything.
        kind, evidence = classify_text("HOW TO MAKE A KURZGESAGT VIDEO")
        assert kind == KIND_LABEL
        assert evidence["finite_verbs"] == []

    def test_short_sentence_is_still_a_sentence(self):
        # The mirror case: two words, but a complete assertion.
        kind, evidence = classify_text("Water boils")
        assert kind == KIND_SENTENCE
        assert evidence["reason"] == "finite_verb"

    def test_content_words_exclude_function_words(self):
        # "Ohm's Law" is two content words, not three.
        _, evidence = classify_text("Ohm's Law")
        assert evidence["content_words"] <= 2

    def test_long_verbless_text_is_still_a_label(self):
        # Defect 10.1 (30 Sep 2026): this used to be the fallback arm's
        # positive case -- no finite verb, but "too long to be a cue" was
        # promoted to sentence anyway. Corpus evidence said otherwise (see
        # onscreen.py's module docstring), so length no longer promotes
        # anything. Nine words, no finite verb, still a label.
        kind, evidence = classify_text(
            "the rapid thermal expansion of superheated metal alloys"
        )
        assert kind == KIND_LABEL
        assert evidence["reason"] == "no_finite_verb"

    def test_evidence_is_recorded_for_audit(self):
        # Where a rater disagrees with the system, the disagreement must be
        # inspectable rather than mysterious.
        _, evidence = classify_text("Mitochondria generate ATP")
        assert set(evidence) >= {"reason", "finite_verbs", "content_words"}

    def test_empty_text_is_a_label(self):
        kind, _ = classify_text("")
        assert kind == KIND_LABEL


class TestOcrNoiseDoesNotFakeSentences:
    """Regression from the real corpus, 09 Sep 2026.

    An animated video scored 13.88 instead of ~0. Both of its two "qualifying"
    windows were misclassifications caused by OCR noise, not by redundancy.
    """

    def test_repeated_label_is_not_promoted_to_a_sentence(self):
        # Captured at three stages of its render, grouped into one element.
        # The tagger read the third repetition as the predicate of the first
        # two and reported a finite verb, turning a topic label into a
        # sentence worth 25.0.
        kind, evidence = classify_text(
            "ELEMENTARY PARTICLES\nELEMENTARY PARTICLES\nELEMENTARYPARTICLES"
        )
        assert kind == KIND_LABEL
        assert evidence["finite_verbs"] == []

    def test_single_character_debris_is_not_a_finite_verb(self):
        # A stray "M" from a graphic was tagged VerbForm=Fin. The shortest real
        # finite verbs in English are two characters.
        kind, evidence = classify_text("M\nM")
        assert kind == KIND_LABEL
        assert evidence["finite_verbs"] == []

    def test_a_verb_alone_is_not_a_clause(self):
        # A predicate needs something to predicate of. Since 05 Oct 2026 a
        # finite verb only counts with a subject, so this is caught there,
        # before the content-word count is ever reached.
        kind, evidence = classify_text("boils")
        assert kind == KIND_LABEL
        assert evidence["finite_verbs"] == []

    def test_genuine_short_sentence_still_passes(self):
        # The guard must not swallow real two-word clauses.
        assert classify_text("Water boils")[0] == KIND_SENTENCE

    def test_genuine_caption_still_passes(self):
        # Real on-screen text from the same video, correctly a sentence.
        assert classify_text("We try\nour best!")[0] == KIND_SENTENCE


class TestDedupeLines:
    def test_exact_repeats_collapse(self):
        assert dedupe_lines("ATP\nATP\nATP") == "ATP"

    def test_spacing_variants_collapse(self):
        # "ELEMENTARYPARTICLES" is the same reading as "ELEMENTARY PARTICLES".
        assert dedupe_lines("ELEMENTARY PARTICLES\nELEMENTARYPARTICLES") == \
            "ELEMENTARY PARTICLES"

    def test_distinct_lines_are_preserved(self):
        out = dedupe_lines("Mitochondria generate ATP\noxidative phosphorylation")
        assert out.count("\n") == 1

    def test_blank_and_punctuation_only_lines_dropped(self):
        assert dedupe_lines("real text\n\n   \n...") == "real text"

    def test_empty(self):
        assert dedupe_lines("") == ""
