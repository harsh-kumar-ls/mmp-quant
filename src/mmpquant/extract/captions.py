"""Whole sentences from captions that run across screens (15 Oct 2026).

Prompted by Microlearning-Safety-Training: its captions show one sentence two
lines at a time over several screens ("Another important section of / the
course focuses" -> "on fire prevention and / emergency preparation."). Each
screen on its own is a fragment with no subject-and-verb, so every piece was
classified a label and the narration reading them aloud word for word scored
as no redundancy at all (1:50-2:00: "no on-screen text").

This module follows the on-screen text through time as one stream and joins
consecutive pieces into sentences. The guards exist because joining is
exactly how the 04 Oct 2026 bug happened (a title and a label shown seconds
apart were glued into a fake sentence, "Gaussian Elimination signed Area"):

  * a piece must look like a caption: typed, clean text -- at least 90% of
    its words real English words (misread chalk never is), no bullet markers
    (a bulleted slide is a list, not running prose), no web addresses
    (watermarks), and not ALL CAPS (titles, brand marks, "SUBSCRIBE");
  * consecutive pieces must sit in the same place on screen (a caption box
    does not move) and follow each other within `max_gap_s`;
  * a joined sentence spans at most `max_pieces` screens;
  * the result must end in . ! or ? and must itself classify as a sentence
    (a finite verb with a subject);
  * only sentences actually formed from two or more pieces are returned --
    single-screen sentences are already found by the per-screen classifier.

A stitched sentence still has to pass the scorer's gate (the narration must
restate it) before it can score anything.

The caption-likeness and same-place guards were added the same day, after the
first corpus run: without them, chalkboard lectures and bulleted slides were
glued into "sentences" ("and dossit qubit is do augthig ...", "Governing
laws/theories •Thermodynamics •Fluid Mechanics ...", "clideo.com ..."), and
test_1 lost a real restated sentence to one of them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..onscreen import KIND_SENTENCE, classify_text, content_lemmas, real_word_share

_TERMINAL = re.compile(r"[.!?][\"'”’)]*$")
_SENT_END = re.compile(r"(?<=[.!?])[\"'”’)]*\s+")
_LOOKBACK = 3
_BULLET = re.compile(r"[•·▪●◦■□►➤✓✔]|^\s*[-*–]\s")
_WEB = re.compile(r"https?://|www\.|\b[\w-]+\.(com|org|net|io|edu|in|co|ly)\b", re.IGNORECASE)
# Notation (an equation is not running prose) and long digit strings (OCR
# noise from a graphic): neither occurs in a spoken-style caption.
_NOTATION = re.compile(r"[=→←↔≠≤≥∑∫√^]|\d{6,}")
# A numbered list ("1. Applied Educational Theory 2. Classroom Management") is a
# list, not running prose -- same principle as bullets (21 Oct 2026: it was
# joined to the next caption sentence in slides_pedagogy_explained).
_ENUMERATOR = re.compile(r"(?:^|\s)\d{1,2}[.)]\s+\S")
MIN_REAL_WORD_SHARE = 0.9

Box = tuple[float, float, float, float] | None


@dataclass
class CaptionPiece:
    text: str
    start: float
    end: float
    block_indices: list[int]
    bbox: Box = None


@dataclass(frozen=True)
class StitchedSentence:
    text: str
    start: float
    end: float
    pieces: tuple[str, ...]
    block_indices: tuple[int, ...]
    # (start, end) of each piece, aligned with `pieces`: which part of the
    # sentence was on screen when (21 Oct 2026, see align._with_stitched).
    spans: tuple[tuple[float, float], ...] = ()

    def visible_text(self, lo: float, hi: float) -> str:
        """The part of the sentence on screen during [lo, hi)."""
        if not self.spans:
            return self.text
        return " ".join(p for p, (a, b) in zip(self.pieces, self.spans, strict=True)
                        if a < hi and b > lo)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def is_caption_like(text: str) -> bool:
    """Typed running prose: not a title, a list item, a watermark, notation or
    misread chalk."""
    if (_is_all_caps(text) or _BULLET.search(text) or _WEB.search(text)
            or _NOTATION.search(text) or len(_ENUMERATOR.findall(text)) >= 2):
        return False
    real, total = real_word_share(text)
    return total > 0 and real / total >= MIN_REAL_WORD_SHARE


def same_place(a: Box, b: Box) -> bool:
    """Whether two text boxes sit in the same place on screen (a caption box):
    they overlap horizontally and their vertical centres are within one and a
    half line-block heights. Unknown positions are not held against a piece."""
    if a is None or b is None:
        return True
    if min(a[2], b[2]) <= max(a[0], b[0]):
        return False
    height = max(a[3] - a[1], b[3] - b[1], 1.0)
    return abs((a[1] + a[3]) - (b[1] + b[3])) / 2 <= 1.5 * height


def caption_stream(
    texts_per_block: Sequence[Sequence[str | tuple[str, Box]]],
    spans: Sequence[tuple[float, float]],
    *,
    similarity: float = 0.88,
    max_gap_s: float = 1.0,
) -> list[CaptionPiece]:
    """On-screen text pieces in time order, each distinct piece once.

    A caption held on screen across several blocks, or captured mid-animation
    ("you unders" -> "you understand s" -> "you understand"), is one piece:
    a reading that is a prefix of, extends, or nearly equals a recent piece
    updates that piece (keeping the longer reading) instead of adding one.
    """
    pieces: list[CaptionPiece] = []
    for b, (texts, (start, end)) in enumerate(zip(texts_per_block, spans, strict=True)):
        for item in texts:
            raw, box = (item, None) if isinstance(item, str) else item
            text = " ".join(raw.split())
            if not text or not is_caption_like(text):
                continue
            k = _key(text)
            if not k:
                continue
            for piece in reversed(pieces[-_LOOKBACK:]):
                if start - piece.end > max_gap_s:
                    continue  # reappearing later is a new piece, not this one
                pk = _key(piece.text)
                if k == pk or pk.startswith(k):
                    break
                if k.startswith(pk) or SequenceMatcher(None, k, pk).ratio() >= similarity:
                    if len(k) > len(pk):
                        piece.text = text
                    break
            else:
                pieces.append(CaptionPiece(text, start, end, [b], box))
                continue
            piece.end = max(piece.end, end)
            if b not in piece.block_indices:
                piece.block_indices.append(b)
    return pieces


def stitch_sentences(
    pieces: Sequence[CaptionPiece], *, max_gap_s: float = 1.0, max_pieces: int = 4
) -> list[StitchedSentence]:
    """Join consecutive pieces into sentences ending in . ! or ?"""
    out: list[StitchedSentence] = []
    buffer: list[CaptionPiece] = []
    for piece in pieces:
        if buffer and (piece.start - buffer[-1].end > max_gap_s
                       or not same_place(buffer[-1].bbox, piece.bbox)):
            buffer = []  # too far apart in time, or not the same caption box
        buffer.append(piece)
        if len(buffer) > max_pieces:
            buffer.pop(0)
        if _TERMINAL.search(piece.text):
            out.extend(_sentences(buffer))
            buffer = []
    return out


def _sentences(buffer: Sequence[CaptionPiece]) -> list[StitchedSentence]:
    joined, offsets = "", []
    for p in buffer:
        if joined:
            joined += " "
        offsets.append((len(joined), len(joined) + len(p.text), p))
        joined += p.text
    out = []
    pos = 0
    for m in list(_SENT_END.finditer(joined)) + [None]:
        stop = m.start() if m else len(joined)
        sentence = joined[pos:stop].strip()
        lo, hi = pos, stop
        pos = m.end() if m else len(joined)
        if not sentence or not _TERMINAL.search(sentence):
            continue
        used = [p for a, b, p in offsets if a < hi and b > lo]
        if len(used) < 2:
            continue  # a one-screen sentence: the per-screen classifier has it
        if len(content_lemmas(sentence)) < 2 or classify_text(sentence)[0] != KIND_SENTENCE:
            continue
        out.append(StitchedSentence(
            text=sentence,
            start=min(p.start for p in used),
            end=max(p.end for p in used),
            pieces=tuple(p.text for p in used),
            block_indices=tuple(sorted({b for p in used for b in p.block_indices})),
            spans=tuple((p.start, p.end) for p in used),
        ))
    return out
