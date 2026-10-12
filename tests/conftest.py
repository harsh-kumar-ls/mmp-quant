"""Shared test fixtures.

The scorer takes its embedder by injection, so every scoring test runs with a
deterministic stand-in and no model download. That is the point of the
`Embedder` protocol: the redundancy formula is testable in milliseconds, and a
formula regression cannot hide behind embedding noise.
"""

from __future__ import annotations

import pytest

from mmpquant.types import Window


class ConstantEmbedder:
    """Returns a fixed cosine similarity, so the verbatim and semantic branches
    of the weighted formula can be isolated from each other."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def encode_pair(self, text_a: str, text_b: str) -> float:
        self.calls += 1
        return self.value


class JaccardEmbedder:
    """Token-Jaccard as a crude semantic proxy — monotonic in overlap, cheap,
    deterministic. Used where a test needs *some* variation rather than a
    specific number."""

    def encode_pair(self, text_a: str, text_b: str) -> float:
        a, b = set(text_a.lower().split()), set(text_b.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


@pytest.fixture
def constant_embedder():
    return ConstantEmbedder


@pytest.fixture
def jaccard_embedder():
    return JaccardEmbedder()


def make_window(
    ocr_text: str,
    asr_text: str,
    index: int = 0,
    start: float = 0.0,
    end: float = 10.0,
    prose_text: str | None = None,
) -> Window:
    """Build a Window directly, bypassing OCR layout reconstruction.

    `prose_text` defaults to the whole of `ocr_text`, i.e. "assume every word on
    screen was classified as prose". That keeps the scoring-formula tests
    focused on the formula. Text classification is exercised separately, in
    test_onscreen.py, against real layout geometry.
    """
    return Window(
        index=index,
        start=start,
        end=end,
        asr_text=asr_text,
        ocr_text=ocr_text,
        ocr_sentence_text=ocr_text if prose_text is None else prose_text,
        asr_pool_start=start - 3.0,
        asr_pool_end=end + 3.0,
        ocr_block_indices=(0,),
    )


# --------------------------------------------------------------------------- #
# Synthetic text fixtures used across the scoring tests.
# --------------------------------------------------------------------------- #

# Narrator reads the slide verbatim: maximum redundancy per Mayer.
REDUNDANT_OCR = "Mitochondria generate ATP through oxidative phosphorylation"
REDUNDANT_ASR = "Mitochondria generate ATP through oxidative phosphorylation"

# Slide shows a diagram caption, narration explains something else entirely:
# well-designed per Mayer, should score near the floor.
CLEAN_OCR = "Figure four shows the electron transport chain across the membrane"
CLEAN_ASR = "Now imagine you are holding a bicycle wheel and you spin it quickly"

# Short signaling label — a key term on screen. NOT redundancy (Mayer's
# signaling exception), must be excluded from scoring entirely.
SHORT_LABEL_OCR = "ATP synthase"
SHORT_LABEL_ASR = "This enzyme sits in the inner membrane and it does the real work here"
