"""The two overlap measures: verbatim (ROUGE-L F1) and semantic (embedding cosine).

Both are implemented here rather than inside the scorer so they can be unit
tested against hand-computed values without loading a video or a model.
"""

from __future__ import annotations

import functools
import itertools
import re
from collections.abc import Sequence
from typing import Protocol

from .textnorm import NormalisationConfig, normalise_for_semantic, normalise_for_verbatim, tokenise

# --------------------------------------------------------------------------- #
# Verbatim: ROUGE-L F1
# --------------------------------------------------------------------------- #


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence.

    Rolling two-row DP: O(len(a) * len(b)) time, O(min) space. Window-sized
    inputs are a few hundred tokens, so this is negligible.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):  # keep the inner loop over the shorter sequence
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(text_a: str, text_b: str, cfg: NormalisationConfig | None = None) -> float:
    """ROUGE-L F1 between two texts, in [0, 1].

    F1 is symmetric in its arguments, so there is no candidate/reference
    ambiguity here — an important property, because "is the slide echoing the
    narration or the narration echoing the slide" is not a question our score
    needs to answer.
    """
    tokens_a = tokenise(normalise_for_verbatim(text_a, cfg))
    tokens_b = tokenise(normalise_for_verbatim(text_b, cfg))
    if not tokens_a or not tokens_b:
        return 0.0
    lcs = lcs_length(tokens_a, tokens_b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(tokens_a)
    recall = lcs / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def rouge_l_screen_coverage(
    screen: str, narration: str, cfg: NormalisationConfig | None = None
) -> float:
    """Share of the on-screen tokens the narration says, in order: LCS / |screen|.

    The redundancy word-match measure since 15 Oct 2026 (user decision). The
    principle asks how much of the on-screen text the narration duplicates.
    ROUGE-L F1 also divides by the narration's length, and the narration pool
    is longer than the screen by construction (the window plus ±3 s, ~16 s of
    speech against one screen of text) -- so a screen read aloud word for
    word scored ~0.5-0.7, never 1 (Microlearning-Safety-Training, 15 Oct).
    """
    tokens_a = tokenise(normalise_for_verbatim(screen, cfg))
    tokens_b = tokenise(normalise_for_verbatim(narration, cfg))
    if not tokens_a or not tokens_b:
        return 0.0
    return lcs_length(tokens_a, tokens_b) / len(tokens_a)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def narration_units(text: str) -> list[str]:
    """Spoken sentences, and each pair of consecutive ones, from narration.

    The unit an on-screen sentence is compared with by meaning (15 Oct 2026):
    the narration sentence that restates it, not the whole window's speech.
    Pairs are included because one caption sentence is often spoken across a
    sentence break the speech recogniser inserted.
    """
    sents = [x.strip() for x in _SENTENCE_END.split(text.strip()) if x.strip()]
    if len(sents) <= 1:
        return [text.strip()] if text.strip() else []
    return sents + [f"{a} {b}" for a, b in itertools.pairwise(sents)]


# --------------------------------------------------------------------------- #
# Semantic: sentence-embedding cosine
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    """Anything that maps a batch of strings to unit-norm vectors."""

    def encode_pair(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity of the two texts."""


class TextEncoder(Protocol):
    """Anything that maps strings to unit-norm vectors (an array, one row each)."""

    def encode(self, texts: Sequence[str]):
        ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers.

    The model (~90 MB for MiniLM) is loaded on first use, not at import, so
    unit tests and `--help` do not pay for it.

    MEASUREMENT CAVEAT worth knowing before interpreting any number: MiniLM
    cosine similarity has a high floor on same-domain English text. Two
    *unrelated* sentences from the same lecture typically sit around 0.2-0.4,
    not 0.0. So the semantic branch contributes a non-trivial baseline even to
    a genuinely non-redundant video, and raw scores are best read as ordinal,
    not absolute. Whether to recalibrate (e.g. affine rescale against an
    empirical non-redundant baseline) is deliberately deferred until the
    professor's scores exist — recalibrating first would be tuning blind.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    @functools.cached_property
    def _model(self):  # pragma: no cover - requires model download
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device=self.device)

    def encode_pair(self, text_a: str, text_b: str) -> float:  # pragma: no cover
        a = normalise_for_semantic(text_a)
        b = normalise_for_semantic(text_b)
        if not a or not b:
            return 0.0
        emb = self._model.encode([a, b], normalize_embeddings=True, show_progress_bar=False)
        return float(emb[0] @ emb[1])

    def encode(self, texts: Sequence[str]):  # pragma: no cover - requires model
        """Unit-norm embeddings, one row per text (coherence: off-topic test)."""
        return self._model.encode(
            [normalise_for_semantic(t) for t in texts],
            normalize_embeddings=True, show_progress_bar=False,
        )


def clamp_similarity(value: float) -> float:
    """Cosine can be negative; a negative overlap is not meaningful here.

    Clamped to [0, 1] so the weighted combination stays in range. This is a
    floor, not a rescale — see the caveat on SentenceTransformerEmbedder.
    """
    return max(0.0, min(1.0, value))
