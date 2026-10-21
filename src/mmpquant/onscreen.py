"""Grouping and classification of on-screen text.

Operative definition
--------------------
From Mayer's Cognitive Theory of Multimedia Learning, as specified by the
supervisor (07 Sep 2026) with supporting literature:

    Redundancy is on-screen text that duplicates concurrent narration.
    Key terms, labels and headlines are *signaling*, not redundancy: they are
    short, incomplete verbal cues that point at what matters. Only complete
    sentences -- text that could substitute for the narration as a self-contained
    explanation -- count as redundancy.

The decisive test is grammatical, not length
--------------------------------------------
Word count alone gets this wrong, and demonstrably so. "HOW TO MAKE A KURZGESAGT
VIDEO" is six words but is a headline, because `make` is an infinitive, not a
finite verb: the phrase states a topic rather than asserting anything. Meanwhile
"Water boils" is two words and is a complete sentence.

Classification therefore follows the operational criteria in the literature:

    finite verb present, >= 2 content words  -> SENTENCE  (redundancy-eligible)
    otherwise                                -> LABEL     (signaling; excluded)

A verbless-but-long branch used to sit between these two ("no finite verb but
enough content words to read as a statement anyway"). It was removed (defect
10.1, 30 Sep 2026): measured against the corpus it never once caught a genuine
verbless declarative, its original justification, and instead classified
title cards, OCR garbage and mathematical notation as sentences purely because
they were long — the opposite of Mayer's signaling exception, which excludes a
title regardless of its length. A sentence now requires a finite verb, with no
length-based exception.

A finite verb is one marked for tense and agreement ("is", "generate", "pumps"),
as opposed to an infinitive ("to make") or a bare participle. spaCy's
morphological analyser reports this directly as VerbForm=Fin, which is what
distinguishes a headline from an assertion without hand-written rules.

Content words exclude determiners, prepositions, conjunctions and punctuation,
so "Ohm's Law" counts as two, not three.

Layout still matters
--------------------
Before anything is classified, OCR lines are grouped into *text elements* by
their position on screen. A wrapped sentence or a bullet block becomes one
element; labels scattered across a frame stay separate. Without this, three
2-word diagram labels are summed into one 6-word "phrase" -- a real defect that
scored an animated video 8.1 when it contained no sentences at all.

Only sentences enter the overlap comparison. Labels are excluded from the
compared text as well as from the qualification decision, because text that
cannot constitute redundancy must not be able to raise the score either.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import re
from dataclasses import dataclass, field

from .textnorm import word_count
from .types import OcrLine

log = logging.getLogger(__name__)

KIND_SENTENCE = "sentence"
KIND_LABEL = "label"

# Parts of speech that carry content. Determiners, adpositions, conjunctions,
# particles and punctuation are excluded, so "Ohm's Law" counts as two content
# words and "In a voltage divider" as two.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


# The parser that decides sentence vs label. en_core_web_trf since 05 Oct 2026:
# on 22 hard cases from the corpus -- 11 headings and labels, 11 real
# sentences -- the small model mistook 4 headings for sentences ("Error
# correction CODES / Hamming codes", "SET theory ... DAT", "Gaussian Elimination
# SIGNED Area"), the medium model 2 plus one real sentence missed, the
# transformer none of either. ~28 ms per text. Set via onscreen.spacy_model,
# which until now was documented in config.yaml but never read.
_SPACY_MODEL = "en_core_web_trf"


def set_spacy_model(name: str) -> None:
    """Choose the parser (from config); takes effect on the next classification."""
    global _SPACY_MODEL
    _SPACY_MODEL = name


def _nlp():
    return _load_nlp(_SPACY_MODEL)


@functools.lru_cache(maxsize=2)
def _load_nlp(model: str):
    """Load the POS tagger once. Returns None if spaCy is unavailable."""
    try:
        import spacy

        return spacy.load(model)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
        # pragma: no cover - environment dependent
        log.warning(
            "spaCy unavailable (%s); falling back to a word-count rule for "
            "sentence/label classification. Headlines with no finite verb will "
            "be misclassified as sentences.",
            exc,
        )
        return None


def dedupe_lines(text: str) -> str:
    """Collapse repeated readings of the same line within one block of text.

    Animated text is captured at several stages of its render, and when those
    captures land in the same screen region they group into a single element
    whose text is one label repeated. That repetition is not harmless: given
    "elementary particles elementary particles elementaryparticles", the tagger
    reads the third repetition as the predicate of the first two and reports a
    finite verb, promoting a topic label to a sentence. Observed on a real
    animated video, where it was the sole cause of the video scoring 13.9
    instead of ~0.
    """
    from .textnorm import normalise_for_verbatim

    # Of two readings where one contains the other, keep the LONGER: it is the
    # more complete render. The first version of this loop kept whichever came
    # first, so a short label above a sentence deleted the sentence -- "Area"
    # is a substring of "areas get scaled by det(A)", and that caption (Test_11)
    # was thrown away as a repeat of the label before it was ever tagged.
    kept: list[tuple[str, str]] = []  # (key, line)
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        key = normalise_for_verbatim(line).replace(" ", "")
        if not key:
            continue
        if any(key == k or key in k for k, _ in kept):
            continue  # already have this reading or a fuller one
        kept = [(k, ln) for k, ln in kept if k not in key]  # drop partials of it
        kept.append((key, line))
    return "\n".join(line for _, line in kept)


def is_real_word(word: str) -> bool:
    """True if `word` exists in English at all, per the wordfreq corpus.

    Defect 10.3 (04 Oct 2026). OCR of chalk handwriting with a lecturer
    standing in front of it returns confident garbage -- "electmmoquehisin"
    for "electromagnetism", "moguehiin", "quantwnmuchanics" -- and the tagger,
    which must assign *some* part of speech to every token, tagged those as
    finite verbs. Garbage was promoted to a sentence and scored (Test_22,
    1:00-1:10: 11.18).

    spaCy cannot catch this itself: the small model ships without vectors, so
    `token.is_oov` is True for every word, real or not. wordfreq's Zipf
    frequency can: measured on the corpus, every real word checked scored above
    2 (including "eigenvalue" 2.38 and "electromagnetism" 2.41) and every piece
    of OCR garbage scored exactly 0.0. The test is "is this a word English uses
    at all" (frequency > 0), not a tuned cutoff, so it does not need
    recalibrating per corpus.
    """
    from wordfreq import zipf_frequency

    return zipf_frequency(word.lower(), "en") > 0


def real_word_share(text: str, *, min_len: int = 3) -> tuple[int, int]:
    """(real, total) counts over the alphabetic words in `text`.

    Only words of `min_len`+ letters are counted: single letters ("x", "A"),
    unit symbols and two-letter fragments are as likely to be legitimate
    notation as noise, and counting them would make every equation look
    garbled. Digits and symbols are skipped for the same reason.
    """
    import re

    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= min_len]
    return sum(1 for w in words if is_real_word(w)), len(words)


# A goal heading at the start of a text element: "Aim:", "Goals -",
# "Learning objectives:", "Outcome:", "Purpose:".
_GOAL_HEADING = re.compile(
    r"^\s*(learning\s+)?(aims?|goals?|objectives?|outcomes?|purpose)\s*[:\-–—]",
    re.IGNORECASE,
)

# Contracted finite verbs, as spaCy tokenises them: "that's" -> "that" + "'s".
# Both apostrophe forms, since OCR returns either.
_CONTRACTED_VERBS = {
    f"{a}{v}" for a in ("'", "\u2019") for v in ("s", "re", "m", "ve", "d", "ll")
}


def _is_plausible_verb(token) -> bool:
    """Reject tags the model had to guess at on OCR noise.

    A finite verb in English is at least two characters ("is", "do", "go"), so a
    lone "M" left over from a graphic is never one. The tagger must assign some
    part of speech to every token, and on single-character debris it guesses.

    It also guesses on longer debris: OCR misreads like "moguehiin" were tagged
    as finite verbs (defect 10.3), so a verb must also be a real English word.

    Contractions are the exception to "alphabetic": in "that's" the finite verb
    is the token "'s", which the alphabetic test always rejected. That went
    unnoticed because noise had been standing in for it -- "But that's a super
    specji r" (Test_12) counted as a sentence only because the garbage token
    "πTT" was tagged as a verb. The caller has already required VerbForm=Fin, so
    a possessive "'s" (tagged PART, not a verb) cannot slip through here.
    """
    if token.text.lower() in _CONTRACTED_VERBS:
        return True
    return token.is_alpha and len(token.text) >= 2 and is_real_word(token.text)


_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}


def _has_subject(token) -> bool:
    """The verb, or the main verb it is an auxiliary of, has a subject."""
    if any(child.dep_ in _SUBJECT_DEPS for child in token.children):
        return True
    if token.dep_ in ("aux", "auxpass"):
        return any(child.dep_ in _SUBJECT_DEPS for child in token.head.children)
    # A verb coordinated with one that has a subject shares it ("Water boils
    # and EVAPORATES").
    if token.dep_ == "conj":
        return _has_subject(token.head)
    return False


def classify_text(text: str, *, min_content_words: int = 5) -> tuple[str, dict]:
    """Classify on-screen text as a sentence or a label, with its evidence.

    The returned evidence is recorded in the audit trail so any classification
    can be checked against a human rater's judgement, which is the whole point
    of writing it down: where the system and a rater disagree, the disagreement
    should be inspectable rather than mysterious.
    """
    stripped = dedupe_lines(text).strip()
    if not stripped:
        return KIND_LABEL, {"reason": "empty", "finite_verbs": [], "content_words": 0}

    # Goal statements are titles (user's ruling, 04 Oct 2026): "Aim: Provide
    # proper mathematical language for classical mechanics, ..." orients the
    # learner the way a title does and places no extra load on them, so it is
    # never redundancy -- whatever its grammar, and even when read aloud.
    # Recognised by an explicit goal heading; an unmarked goal statement
    # ("Today we will learn ...") is not caught by this rule.
    if _GOAL_HEADING.match(stripped):
        return KIND_LABEL, {"reason": "goal_statement", "finite_verbs": [], "content_words": 0}

    # OCR frequently returns slide text in full caps, which degrades tagging.
    probe = stripped.lower() if stripped.isupper() else stripped
    probe = probe.replace("\n", " ")

    nlp = _nlp()
    if nlp is None:
        n = word_count(stripped)
        kind = KIND_SENTENCE if n >= min_content_words else KIND_LABEL
        return kind, {
            "reason": "word_count_fallback_no_tagger",
            "finite_verbs": [],
            "content_words": n,
        }

    doc = nlp(probe)
    # A finite verb counts only with a subject: something it is said of
    # ("Water BOILS", "areas GET scaled"). This is the grammatical content of
    # the old "two content words: a subject and a verb" rule, checked on the
    # parse instead of counted.
    finite = [
        t.text
        for t in doc
        if "Fin" in t.morph.get("VerbForm") and _is_plausible_verb(t) and _has_subject(t)
    ]
    content = sum(1 for t in doc if t.pos_ in _CONTENT_POS)

    # A finite verb alone is not a sentence: a predicate needs something to
    # predicate of. Two content words is the minimum that can carry a subject
    # and a verb ("Water boils"), and it excludes single-token OCR debris that
    # happened to be tagged as a verb.
    #
    # Defect 10.1 (30 Sep 2026): a third branch used to exist here --
    # `elif content >= min_content_words: kind = SENTENCE` -- for a verbless
    # statement long enough to read as a claim anyway. Measured against the
    # corpus, that branch caught not one genuine example of that. It caught,
    # every time it was inspected: a university title card ("GEOMETRIC ANATOMY
    # OF THEORETICALPHYSICS ... Frederic P.", 11 words, no verb), OCR garbage
    # that happened to tokenize into enough fragments ("Lana Lemy lemgf lengfh
    # lenyfh lenyfh m"), and mathematical notation ("3c + 2y = -4"). All three
    # are exactly what Mayer's signaling exception is for -- a title names a
    # topic regardless of its length, and an equation is notation, not prose --
    # and none of them can duplicate narration in the sense the principle
    # describes. Across the full 24-video corpus, 90 of 228 qualifying windows
    # (39%) rested entirely on this branch; sampled across a dozen of them, the
    # branch never once caught a real verbless declarative statement, its
    # original justification. `min_content_words` is kept as a parameter for
    # compatibility but no longer used to promote text to a sentence on length
    # alone -- see PROGRESS.md, defect 10.1.
    if finite and content >= 2:
        reason = "finite_verb"
        kind = KIND_SENTENCE
    elif finite:
        reason = "finite_verb_but_too_few_content_words_to_be_a_clause"
        kind = KIND_LABEL
    else:
        reason = "no_finite_verb"
        kind = KIND_LABEL

    return kind, {
        "reason": reason,
        "finite_verbs": finite,
        "content_words": content,
    }


@dataclass(frozen=True)
class TextElement:
    """A run of OCR lines that form one block of text on screen."""

    lines: tuple[OcrLine, ...]
    kind: str
    bbox: tuple[float, float, float, float] | None
    evidence: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def word_count(self) -> int:
        return word_count(self.text)

    @property
    def is_sentence(self) -> bool:
        return self.kind == KIND_SENTENCE

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "word_count": self.word_count,
            "n_lines": len(self.lines),
            "bbox": list(self.bbox) if self.bbox else None,
            **self.evidence,
        }


def _height(line: OcrLine) -> float:
    if not line.bbox:
        return 0.0
    return abs(line.bbox[3] - line.bbox[1])


def _belongs_to_same_element(
    prev: OcrLine,
    curr: OcrLine,
    *,
    line_gap_ratio: float,
    x_align_tol_px: float,
) -> bool:
    """Whether `curr` continues the text block that `prev` belongs to.

    Two conditions, both required:

      * Vertically adjacent. The gap between the bottom of one line and the top
        of the next must be small relative to the line height. A wrapped
        sentence has almost no gap; two unrelated captions on opposite halves of
        a diagram have a large one.
      * Horizontally related. Either their left edges line up (a bullet list or
        paragraph is left-aligned) or their horizontal spans overlap
        substantially (a centred title wrapping onto a second line).

    Without bounding boxes there is no layout to reason about, so lines are kept
    separate. That is the conservative choice: it can split a real sentence into
    two short labels and under-report redundancy, which is preferable to
    fusing unrelated labels into fictitious prose and over-reporting it.
    """
    if not prev.bbox or not curr.bbox:
        return False

    px0, _py0, px1, py1 = prev.bbox
    cx0, cy0, cx1, _cy1 = curr.bbox

    gap = cy0 - py1
    # The two lines' mean height, not the larger (21 Oct 2026): OCR once drew
    # a London subtitle line's box 96 px tall for 50 px text, and against
    # that one box the 129 px gap up to the watermark looked like line spacing.
    ref_height = max((_height(prev) + _height(curr)) / 2, 1.0)
    if gap > line_gap_ratio * ref_height:
        return False
    if gap < -ref_height:  # wildly overlapping: different regions, not a block
        return False

    left_aligned = abs(cx0 - px0) <= x_align_tol_px
    overlap = min(px1, cx1) - max(px0, cx0)
    narrower = max(min(px1 - px0, cx1 - cx0), 1.0)
    horizontally_related = left_aligned or (overlap / narrower) >= 0.5
    return horizontally_related


def group_lines(
    lines: tuple[OcrLine, ...] | list[OcrLine],
    *,
    line_gap_ratio: float = 1.6,
    x_align_tol_px: float = 24.0,
) -> list[list[OcrLine]]:
    """Group OCR lines (in reading order) into the text blocks they form on
    screen, by layout alone -- no classification.

    A lone glyph -- a line with fewer than two letters or digits -- is a group
    of its own and never links the lines around it (21 Oct 2026). It carries
    no words, and as a bridge it joins things that are far apart: on London at
    1:10 a stray "L" (part of the picture) sat between the watermark
    "(c) Loescher Editore" and a subtitle, and chained them into one
    "sentence"; without it they are 3.7 line heights apart.
    """
    lines = [line for line in lines if line.text.strip()]
    groups: list[list[OcrLine]] = []
    current: list[OcrLine] | None = None  # the group of the last real line
    for line in lines:
        if _is_lone_glyph(line.text):
            groups.append([line])
            continue
        if current is not None and _belongs_to_same_element(
            current[-1], line, line_gap_ratio=line_gap_ratio, x_align_tol_px=x_align_tol_px
        ):
            current.append(line)
        else:
            current = [line]
            groups.append(current)
    return groups


def _is_lone_glyph(text: str) -> bool:
    return sum(c.isalnum() for c in text) < 2


def group_into_elements(
    lines: tuple[OcrLine, ...] | list[OcrLine],
    *,
    min_prose_words: int = 5,
    line_gap_ratio: float = 1.6,
    x_align_tol_px: float = 24.0,
) -> list[TextElement]:
    """Reconstruct text elements from OCR lines and classify each one.

    Lines are expected in reading order (top-to-bottom, left-to-right), which is
    how the OCR backend returns them.
    """
    groups = group_lines(lines, line_gap_ratio=line_gap_ratio, x_align_tol_px=x_align_tol_px)
    elements: list[TextElement] = []
    for group in groups:
        text = "\n".join(line.text for line in group)
        kind, evidence = classify_text(text, min_content_words=min_prose_words)
        elements.append(
            TextElement(
                lines=tuple(group),
                kind=kind,
                bbox=_union_bbox(group),
                evidence=evidence,
            )
        )
    return elements


def _union_bbox(lines: list[OcrLine]) -> tuple[float, float, float, float] | None:
    boxes = [line.bbox for line in lines if line.bbox]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def merge_spans(spans) -> list[list[float]]:
    """Union of [start, end) intervals, sorted and merged."""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def collapse_near_duplicates(
    elements: list[TextElement], *, similarity_threshold: float = 0.88
) -> list[TextElement]:
    """Drop elements that are repeat readings of text already accounted for.

    Necessary because animated text is captured mid-render. A real title card
    animating in over two seconds was read at 2 fps as "HOWTOMAKEA", then
    "HOWTOI MAKEA", then "HOWTO MAKEA", then "HOW TO MAKE A". Exact-match
    de-duplication keeps all four, and one six-word title is counted as twenty
    words.

    Three tests are applied to whitespace-stripped normalised text, in order of
    strictness:

      * equality, which catches a static label re-read on every frame;
      * containment, which catches partial renders of a longer string;
      * character similarity above `similarity_threshold`, which catches OCR
        corruption during the render -- "howtoimakea" is neither equal to nor
        contained in "howtomakea", because of one spurious character, but is
        95% identical to it.

    The similarity test is deliberately strict. At 0.88 it collapses readings of
    the same text, while leaving genuinely different short labels apart; two
    unrelated labels of similar length are nowhere near 88% character-identical.
    Longer variants win, being the more complete reading -- except among
    near-copies (equal, or above `similarity_threshold`), where the reading on
    screen longest wins (21 Oct 2026). Those are the same text read
    differently, and the reading seen for most of its time is the likeliest
    correct one: a London subtitle read "...you can walk" for 3.5 s and
    "...you can walk." (a stray full stop) for 0.5 s was kept with the full
    stop, split in two there, and lost its second half.
    """
    from difflib import SequenceMatcher

    from .textnorm import normalise_for_verbatim

    ranked = sorted(elements, key=lambda e: -len(e.text))
    # Per kept element: its key, the element, the kinds of every reading
    # that is a near-copy of it (equal, or above `similarity_threshold`), and
    # the time spans of every reading of it (21 Oct 2026: how long the text
    # is on screen decides how much it counts).
    kept: list[tuple[str, list[TextElement], list[str], list]] = []
    for element in ranked:
        key = normalise_for_verbatim(element.text).replace(" ", "")
        if not key:
            continue
        duplicate = False
        spans = list(element.evidence.get("spans") or [])
        for seen, readings, votes, seen_spans in kept:
            if key == seen or SequenceMatcher(None, key, seen).ratio() >= similarity_threshold:
                votes.append(element.kind)
                seen_spans.extend(spans)
                readings.append(element)
                duplicate = True
                break
            if key in seen:
                duplicate = True
                # A partial render of this text (a caption caught mid-draw) is
                # the same text on screen; a short label that merely occurs
                # inside a longer text is not, so it lends no screen time.
                if len(key) * 2 >= len(seen):
                    seen_spans.extend(spans)
                break
        if not duplicate:
            kept.append((key, [element], [element.kind], spans))

    def most_seen(readings: list[TextElement]) -> TextElement:
        # total screen time of each distinct reading; ties (e.g. no spans
        # recorded) -> the longer text, then the first
        time: dict[str, float] = {}
        for e in readings:
            text = " ".join(e.text.split())
            time[text] = time.get(text, 0.0) + sum(b - a for a, b in e.evidence.get("spans") or [])
        return max(readings, key=lambda e: (round(time[" ".join(e.text.split())], 3), len(e.text)))

    out = []
    for _key, readings, votes, spans in kept:
        element = most_seen(readings)
        if spans:
            element = dataclasses.replace(
                element, evidence={**element.evidence, "spans": merge_spans(spans)})
        # A filmed board is read ~20 times per 10 s window, and with chalk some
        # reading always has a misread that looks like a verb ("for" -> "fer",
        # Test_22). One reading in twenty must not decide what the text is: it
        # is a sentence only if most full readings of it say so.
        majority = KIND_SENTENCE if votes.count(KIND_SENTENCE) * 2 > len(votes) else KIND_LABEL
        if majority != element.kind:
            element = dataclasses.replace(
                element,
                kind=majority,
                evidence={**element.evidence, "reason": "majority_of_readings",
                          "readings": len(votes),
                          "readings_as_sentence": votes.count(KIND_SENTENCE)},
            )
        out.append(element)

    order = {id(e): i for i, e in enumerate(elements)}
    for readings, final in zip([k[1] for k in kept], out, strict=True):
        order[id(final)] = min(order[id(r)] for r in readings)
    return sorted(out, key=lambda e: order[id(e)])


def sentence_text(elements: list[TextElement]) -> str:
    """The on-screen text eligible for the overlap comparison.

    Only sentences. Labels and headlines are excluded from the compared text,
    not just from the qualification decision -- under the operative definition
    they cannot constitute redundancy, so they must not be able to raise the
    score either.
    """
    return "\n".join(e.text for e in elements if e.is_sentence)


# Retained under the previous name so existing callers keep working.
prose_text = sentence_text


# --------------------------------------------------------------------------- #
# Sentence-level comparison (05 Oct 2026)
# --------------------------------------------------------------------------- #

# A bullet or an enumerator starts a new sentence: "•", "▪", "1)", "(a)", "6a)",
# "2." -- slide bullets carry no full stop, and spaCy would otherwise read a
# whole bulleted slide as one sentence.
_ITEM_MARKER = re.compile(
    r"(?:^|\s)(?:[•▪◦●‣*]|\(?[0-9]{0,2}[a-z]?\)|[0-9]+\.(?=\s))\s+",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=50_000)
def content_lemmas(text: str) -> frozenset[str]:
    """The content words of `text` as lemmas ("statements" -> "statement").

    Stop words are dropped; words need 3+ letters, numbers 2+ characters
    ("25", "300", "0.25") -- the same rule as the redundancy scorer's content
    words, after the same normalisation, so OCR and narration are comparable.
    """
    from spacy.lang.en.stop_words import STOP_WORDS

    from .textnorm import normalise_for_verbatim

    nlp = _nlp()
    norm = normalise_for_verbatim(text)
    tokens = nlp(norm) if nlp is not None else []
    out = set()
    for t in tokens:
        lemma = t.lemma_.lower()
        if t.text in STOP_WORDS or lemma in STOP_WORDS:
            continue
        if len(lemma) >= 3 or (len(lemma) >= 2 and any(c.isdigit() for c in lemma)):
            out.add(lemma)
    return frozenset(out)


@functools.lru_cache(maxsize=50_000)
def content_forms(text: str) -> frozenset[tuple[str, str]]:
    """(lemma, surface) for every content word -- same filtering as
    `content_lemmas`. Lets a word count as said when EITHER form matches:
    the tagger lemmatises one word differently by context ("not prepared":
    adjective -> "prepared" in the narration, verb -> "prepare" on screen),
    and Test_28 lost a word read aloud verbatim to that (21 Oct 2026)."""
    from spacy.lang.en.stop_words import STOP_WORDS

    from .textnorm import normalise_for_verbatim

    nlp = _nlp()
    tokens = nlp(normalise_for_verbatim(text)) if nlp is not None else []
    out = set()
    for t in tokens:
        lemma, surface = t.lemma_.lower(), t.text.lower()
        if t.text in STOP_WORDS or lemma in STOP_WORDS:
            continue
        if len(lemma) >= 3 or (len(lemma) >= 2 and any(c.isdigit() for c in lemma)):
            out.add((lemma, surface))
    return frozenset(out)


def said_forms(text: str) -> frozenset[str]:
    """Every lemma and every surface form of the content words in `text`,
    plus each pair of adjacent words written as one (21 Oct 2026): the speech
    recogniser splits compounds the screen writes as one word -- "micro
    -learning" for "Microlearning" (user: "the transcript is reading it
    wrong"), "work places" for "workplaces"."""
    from .textnorm import normalise_for_verbatim

    tokens = [t for t in normalise_for_verbatim(text).split() if t.isalpha()]
    joined = {a + b for a, b in itertools.pairwise(tokens) if len(a) >= 2 and len(b) >= 2}
    return frozenset(f for pair in content_forms(text) for f in pair) | joined


def question_said_forms(text: str) -> frozenset[str]:
    """`said_forms` of only the sentences asked as questions (see question_lemmas)."""
    nlp = _nlp()
    sents = [s.text.strip() for s in nlp(text).sents] if nlp is not None else [text]
    out: set[str] = set()
    for s in sents:
        if s.endswith("?"):
            out |= said_forms(s)
    return frozenset(out)


def shared_content(sentence: str, said: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(the sentence's content lemmas, those said): a word counts as said if
    its lemma or its surface form is among the narration's forms."""
    pairs = content_forms(sentence)
    words = frozenset(lemma for lemma, _ in pairs)
    shared = frozenset(lemma for lemma, surface in pairs if lemma in said or surface in said)
    return words, shared


@functools.lru_cache(maxsize=50_000)
def onscreen_sentences(text: str, min_content_words: int = 2) -> tuple[str, ...]:
    """The individual sentences in a block of on-screen text.

    Split at bullets and enumerators, then at sentence boundaries; only pieces
    that are sentences in their own right (finite verb with a subject) and
    carry at least `min_content_words` content words are returned. A heading,
    an author line or a journal name inside a paragraph is not a sentence and
    cannot be "restated".
    """
    nlp = _nlp()
    out: list[str] = []
    for piece in _ITEM_MARKER.split(text.replace("\n", " ")):
        piece = piece.strip()
        if not piece:
            continue
        parts = [s.text.strip() for s in nlp(piece).sents] if nlp is not None else [piece]
        for s in parts:
            if (
                s
                and len(content_lemmas(s)) >= min_content_words
                and classify_text(s)[0] == KIND_SENTENCE
            ):
                out.append(s)
    return tuple(dict.fromkeys(out))


_QUESTION_START = re.compile(
    r"^\W*(?:\(?[0-9]{0,2}[a-z]?\)\s*)?(what|how|why|when|where|which|who|whom|whose)\b",
    re.IGNORECASE,
)


_INVERTED = re.compile(
    r"^\W*(?:\(?[0-9]{0,2}[a-z]?\)\s*)?(what|how|why|when|where|which|who|whom|whose)\s+"
    r"(is|are|was|were|am|do|does|did|can|could|will|would|shall|should|"
    r"may|might|must|has|have|had|happens|happened)\b",
    re.IGNORECASE,
)


def is_question(sentence: str) -> bool:
    """An on-screen question: ends with "?", or opens with a question word
    followed directly by a verb ("What is the new volume ..." with the "?"
    lost by OCR).

    Until 21 Oct 2026 any sentence OPENING with a question word counted, so
    "When it opened on 10th June, the Bridge had to be closed ..." -- a
    statement -- was compared only with narration spoken as a question and
    scored as never said (animation_london_subtitled, 1:27), though it was
    read aloud word for word.
    """
    s = sentence.strip()
    return s.endswith("?") or bool(_INVERTED.match(s))


@functools.lru_cache(maxsize=50_000)
def question_lemmas(text: str) -> frozenset[str]:
    """Content lemmas of only the sentences in `text` that are asked as questions.

    Whisper punctuates spoken questions, so "Now how much water was removed?"
    ends with "?" and "So that's the new volume of the solution." does not.
    """
    nlp = _nlp()
    sents = [s.text.strip() for s in nlp(text).sents] if nlp is not None else [text]
    out: set[str] = set()
    for s in sents:
        if s.endswith("?"):
            out |= content_lemmas(s)
    return frozenset(out)
