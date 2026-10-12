"""Split words that OCR ran together, and nothing else.

Defect 10.4 (04 Oct 2026). On some printed slides the OCR reads every letter
correctly but drops the spaces: "Heatcannotspontaneouslyflowfromacolderbody"
(test_1). One "word" hides the sentence's verb from the classifier and shares
no token with the narration, so a sentence the professor reads aloud word for
word scores as no overlap at all.

The split is the most probable sequence of English words under wordfreq's
frequencies -- the same vocabulary `onscreen.is_real_word` uses. The hard part
is not splitting; it is refusing to split garbage. A frequency list that
contains every fragment of the web will split anything: chalk misreads come
apart as "mogu eh i in" and, worse, "elect mmo que his in" -- which contains a
real verb. So a split is accepted only if every piece is believable, and how
believable a piece has to be depends on its length:

    1 letter   -> only "a" or "i"
    2 letters  -> only a closed list of English two-letter function words
    3-4        -> a common word (Zipf >= 4, i.e. >= 10 per million words)
    5+         -> any word English uses at all (Zipf > 0)

The reasoning is chance: the shorter a string, the more likely it turns up
inside random letters, so the more evidence it needs before being believed.
Checked against 16 run-together phrases and 18 garbage tokens from the corpus:
every phrase in plain English split correctly; every garbage token was refused;
two Latin species names were refused (no word list can split them); one split
is wrong ("PowersofTen" -> "powers often"). The rule was designed and checked
on those same examples, so it is not validated on held-out data -- see
PROGRESS.md. The scoring gate (narration must actually say on-screen content
words) is the backstop: a wrong split cannot score unless the narrator says
the words it produced.
"""

from __future__ import annotations

import functools
import itertools
import math
import re

# Tokens shorter than this are not treated as run-together text: two English
# words need at least that many letters to be worth splitting reliably.
MIN_TOKEN_LEN = 6
MAX_WORD_LEN = 20
SHORT_WORD_MIN_ZIPF = 4.0

_TWO_LETTER_WORDS = frozenset({
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "oh", "on", "or", "so", "to", "up", "us", "we",
})

_ALPHA_RUN = re.compile(r"[A-Za-z]+")


@functools.lru_cache(maxsize=200_000)
def _cost(word: str) -> float:
    from wordfreq import word_frequency

    f = word_frequency(word, "en")
    return math.inf if f <= 0 else -math.log(f)


@functools.lru_cache(maxsize=200_000)
def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, "en")


def _believable(piece: str) -> bool:
    if len(piece) == 1:
        return piece in ("a", "i")
    if len(piece) == 2:
        return piece in _TWO_LETTER_WORDS
    if len(piece) <= 4:
        return _zipf(piece) >= SHORT_WORD_MIN_ZIPF
    return _zipf(piece) > 0


def _segment(lower: str) -> list[int] | None:
    """Split points of the most probable word sequence, or None."""
    n = len(lower)
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_WORD_LEN), i):
            c = best[j] + _cost(lower[j:i])
            if c < best[i]:
                best[i], back[i] = c, j
    if best[n] == math.inf:
        return None
    cuts, i = [], n
    while i > 0:
        cuts.append(i)
        i = back[i]
    return sorted(cuts)[:-1]  # interior split points only


def split_token(token: str) -> str:
    """`token` with spaces inserted, or unchanged if no believable split."""
    from .onscreen import is_real_word

    if len(token) < MIN_TOKEN_LEN or is_real_word(token):
        return token
    lower = token.lower()
    cuts = _segment(lower)
    if not cuts:
        return token
    bounds = [0, *cuts, len(token)]
    pieces = [lower[a:b] for a, b in itertools.pairwise(bounds)]
    if not all(_believable(p) for p in pieces):
        return token
    # Text that "ran together" holds at least two substantive words. One long
    # word plus short fragments is far likelier a misread of one word:
    # "electromagnetisin" (chalk, Test_22) would otherwise become
    # "electromagnet is in" -- inventing the finite verb "is".
    if sum(1 for p in pieces if len(p) >= 3) < 2:
        return token
    # Insert spaces into the original string, preserving its letter case.
    return " ".join(token[a:b] for a, b in itertools.pairwise(bounds))


def split_run_together(text: str) -> str:
    """Apply `split_token` to every alphabetic run in `text`."""
    return _ALPHA_RUN.sub(lambda m: split_token(m.group(0)), text)

