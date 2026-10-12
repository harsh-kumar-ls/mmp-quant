"""Text normalisation shared by the OCR and ASR sides.

Why this module exists
----------------------
Verbatim overlap (ROUGE-L) compares two token sequences that come from very
different generators, and comparing them raw systematically *underestimates*
redundancy:

  * OCR wraps and hyphenates:  "photo-\\nsynthesis"  vs  ASR "photosynthesis"
  * OCR keeps slide casing:    "MITOCHONDRIA"        vs  ASR "mitochondria"
  * OCR reads numerals:        "3 stages"            vs  ASR "three stages"
  * OCR emits bullets/glyphs:  "• ATP  —"            vs  ASR nothing

So a video where the narrator reads the slide word-for-word — maximum
redundancy — would score well below 1.0 on raw text. That is a measurement
error, not a property of the video.

Two normalisation levels are exposed deliberately:

  `normalise_for_verbatim()`  aggressive; feeds ROUGE-L token matching.
  `normalise_for_semantic()`  light; feeds the sentence embedder, which is
                              trained on natural text and performs *worse* on
                              lowercased, de-punctuated input.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# number words -> digits, applied to both sides so "3" == "three"
# --------------------------------------------------------------------------- #

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS: dict[str, str] = {w: str(v) for w, v in {**_UNITS, **_TENS}.items()}

# Unicode punctuation/symbol categories to strip, plus bullets ffmpeg-sampled
# slides are full of.
_KEEP_ALNUM_SPACE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_LINEBREAK_HYPHEN = re.compile(r"(\w)[-‐‑‒–]\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")

# Numbers (04 Oct 2026). Read-aloud content in a chemistry or maths video is
# largely numbers, and they never matched: "0.25" was split into "0" and "25"
# by punctuation stripping, units stayed glued on ("0.25M" -> "25m",
# "300mL" -> "300ml") so they never equalled the spoken "0.25" or "300", and
# Whisper writes decimals with a stray space ("0 .25"). All three are undone
# here, identically for OCR and ASR text.
_ASR_DECIMAL_GAP = re.compile(r"(?<=\d)\s+\.(?=\d)")  # "0 .25" -> "0.25"
_NUMBER_THEN_LETTERS = re.compile(r"(?<=\d)(?=[^\W\d_])")  # "300ml" -> "300 ml"
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_DECIMAL_PLACEHOLDER = re.compile(r"(?<=\d)_(?=\d)")


@dataclass(frozen=True)
class NormalisationConfig:
    lowercase: bool = True
    dehyphenate_linebreaks: bool = True
    strip_punctuation: bool = True
    words_to_digits: bool = True
    collapse_whitespace: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> NormalisationConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _nfkc(text: str) -> str:
    """Fold ligatures, full-width forms and lookalike glyphs to canonical forms.

    OCR on slides routinely emits U+FB01 (fi ligature) and full-width digits;
    NFKC makes those compare equal to what ASR produces.
    """
    return unicodedata.normalize("NFKC", text)


def normalise_for_semantic(text: str, cfg: NormalisationConfig | None = None) -> str:
    """Light clean-up only: fix OCR line-wrapping artifacts, keep case + punctuation.

    Sentence-transformer models are trained on natural text. Stripping case and
    punctuation measurably degrades their embeddings, so the semantic branch
    gets a gentler pipeline than the verbatim branch.
    """
    cfg = cfg or NormalisationConfig()
    out = _nfkc(text)
    if cfg.dehyphenate_linebreaks:
        out = _LINEBREAK_HYPHEN.sub(r"\1\2", out)
    out = out.replace("\n", " ")
    if cfg.collapse_whitespace:
        out = _WHITESPACE.sub(" ", out)
    return out.strip()


def normalise_for_verbatim(text: str, cfg: NormalisationConfig | None = None) -> str:
    """Aggressive normalisation for token-level (ROUGE-L) comparison."""
    cfg = cfg or NormalisationConfig()
    out = _nfkc(text)
    if cfg.dehyphenate_linebreaks:
        out = _LINEBREAK_HYPHEN.sub(r"\1\2", out)
    out = out.replace("\n", " ")
    out = _ASR_DECIMAL_GAP.sub(".", out)
    if cfg.lowercase:
        out = out.lower()
    out = _NUMBER_THEN_LETTERS.sub(" ", out)
    if cfg.strip_punctuation:
        # Do this before word->digit so "three," maps too. A decimal point is
        # kept: underscore survives the strip (it is a word character) and is
        # turned back into a point straight after.
        out = _DECIMAL_POINT.sub("_", out)
        out = _KEEP_ALNUM_SPACE.sub(" ", out)
        out = _DECIMAL_PLACEHOLDER.sub(".", out)
    if cfg.collapse_whitespace:
        out = _WHITESPACE.sub(" ", out)
    out = out.strip()
    if cfg.words_to_digits:
        out = " ".join(_NUMBER_WORDS.get(tok, tok) for tok in out.split())
    return out


def tokenise(text: str) -> list[str]:
    """Whitespace tokenisation. Call on already-normalised text."""
    return text.split()


def word_count(text: str) -> int:
    """Word count used for the signaling-exception threshold.

    Counts on *verbatim-normalised* text so that bullets, dashes and stray OCR
    glyphs do not inflate a 3-word slide title past the 5-word threshold and
    wrongly turn a signaling cue into a redundancy hit.
    """
    return len(tokenise(normalise_for_verbatim(text)))
