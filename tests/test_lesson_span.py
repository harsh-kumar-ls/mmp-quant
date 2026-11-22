"""Intro and outro detection (user's ruling, 13 Nov 2026; lesson_span.py)."""

from __future__ import annotations

from mmpquant.lesson_span import lesson_span, within
from mmpquant.types import AsrSegment, AsrWord, Transcript


def _tr(sentences):
    """sentences: (start, end, text) -> one segment each, words spread evenly."""
    segs = []
    for start, end, text in sentences:
        toks = text.split()
        step = (end - start) / len(toks)
        words = tuple(AsrWord(start + i * step, start + (i + 1) * step, t)
                      for i, t in enumerate(toks))
        segs.append(AsrSegment(start, end, text, words))
    return Transcript(tuple(segs), "en", 0.99, 100.0)


def test_intro_is_before_the_first_word_and_outro_after_the_last():
    sp = lesson_span(_tr([(15.3, 30, "Every day we face small safety risks."),
                          (30, 80, "Heat stress causes dizziness.")]), 100.0)
    assert (sp.start, sp.end) == (15.3, 80) and sp.intro_s == 15.3 and sp.outro_s == 20
    assert sp.closing == ()


def test_the_closing_sign_off_is_outro():
    # Test_28's ending (13 Nov 2026)
    sp = lesson_span(_tr([
        (0, 60, "Heat stress causes dizziness."),
        (60, 70, "Enroll today in the Micro Learning Safety Everywhere course."),
        (70, 80, "Click the link in the description to begin your training now."),
    ]), 90.0)
    assert sp.end == 60 and len(sp.closing) == 2


def test_the_sign_off_run_stops_at_the_first_teaching_sentence():
    sp = lesson_span(_tr([
        (0, 50, "Thanks for watching is what some teachers say first."),
        (50, 60, "Water boils at one hundred degrees."),
        (60, 70, "See you guys next time."),
    ]), 70.0)
    assert sp.end == 60 and sp.closing == ("See you guys next time.",)


def test_a_video_that_is_all_sign_off_is_scored_whole():
    sp = lesson_span(_tr([(0, 10, "Subscribe for more videos.")]), 12.0)
    assert (sp.start, sp.end, sp.closing) == (0, 10, ())


def test_no_speech_gives_no_span():
    assert lesson_span(Transcript((), "en", 0.9, 30.0), 30.0) is None


def test_within_keeps_only_the_lessons_words():
    tr = _tr([(0, 60, "Heat stress causes dizziness."),
              (60, 70, "Click the link in the description now.")])
    kept = within(tr, lesson_span(tr, 70.0))
    assert " ".join(w.text for w in kept.words) == "Heat stress causes dizziness."
