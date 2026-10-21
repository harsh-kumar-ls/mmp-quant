"""Mayer's Redundancy Principle scorer.

Operative definition (supervisor, 07 Sep 2026, with cited literature):

    Redundancy is on-screen text that duplicates concurrent narration, splitting
    the viewer's attention between reading and listening. Key terms, labels and
    headlines are *signaling*, not redundancy.

Only text classified as a SENTENCE enters this scorer. Classification happens
upstream in `mmpquant.onscreen`, and turns on grammatical completeness rather
than length: text carrying a finite verb asserts something and could substitute
for the narration, whereas a headline names a topic. Labels are removed from the
compared text entirely, so they cannot raise the score even indirectly.

    Sentence-level gate (04-05 Oct 2026): a window scores 0 unless the
    narration restates an on-screen sentence -- says more than
    `min_sentence_coverage` of its content words (at least
    `min_shared_content_words`), and more than narration elsewhere in the same
    video says of that sentence. Text on screen is not redundancy by being
    there, and words gathered from all over a screen are not a sentence.

    Above the gate, both measures are corrected for chance, against the same
    on-screen text compared with narration from other moments of the video
    (more than `null_exclusion_s` away):

        v = chance_corrected(ROUGE-L F1 over content words)
        s = chance_corrected(cosine)
        chance_corrected(x) = max(0, (x - median_elsewhere) / (1 - median_elsewhere))

    window_redundancy = 100 * (w_v * v + w_s * s)

    A window is scored only if it holds at least `min_onscreen_words` words of
    sentence text and at least `min_narration_words` of narration.

    result.score = duration-weighted mean over qualifying windows (the
                   "assessed-only" average). The pipeline reports instead the
                   whole-video mean in which an unassessed window counts 0
                   (see mmpquant.segments, "One averaging rule").

A window with no sentence text scores nothing and is excluded from the mean; a
video or segment with no qualifying window at all scores 0, since it was
assessed and found to contain no redundancy. N/A is reserved for the case where
no narration exists anywhere, which makes the principle unevaluable.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import Any

from spacy.lang.en.stop_words import STOP_WORDS  # a word list; loads no model

from ..onscreen import (
    content_lemmas,
    is_question,
    onscreen_sentences,
    question_said_forms,
    said_forms,
    shared_content,
)
from ..overlap import (
    Embedder,
    clamp_similarity,
    narration_units,
    rouge_l_f1,
    rouge_l_screen_coverage,
)
from ..textnorm import NormalisationConfig, normalise_for_verbatim, word_count
from ..types import Window, WindowScore
from .base import PrincipleScorer, register

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Two showings of the same sentence this close together are one showing: a
# subtitle re-detected after a background change, or cut at a block edge.
_SAME_SHOWING_GAP_S = 1.0


def _showing_key(sentence: str) -> str:
    return normalise_for_verbatim(sentence).replace(" ", "")

TIME_STEP_S = 0.5  # the time grid of the window score: the frame sampling interval


def _above_chance(value: float, chance: float) -> float:
    """How far `value` sits above `chance`, as a share of the room above it."""
    if chance >= 1.0:
        return 0.0
    return max(0.0, (value - chance) / (1.0 - chance))


EXCL_NO_ONSCREEN_TEXT = "no_onscreen_text"
EXCL_SIGNALING_EXCEPTION = "signaling_exception_short_onscreen_text"
EXCL_NO_NARRATION = "no_narration"


@register
class RedundancyScorer(PrincipleScorer):
    name = "redundancy"

    def __init__(
        self,
        embedder: Embedder,
        *,
        weight_verbatim: float = 0.6,
        weight_semantic: float = 0.4,
        min_onscreen_words: int = 5,
        min_narration_words: int = 1,
        norm: NormalisationConfig | None = None,
        embedding_model_name: str = "unknown",
        min_shared_content_words: int = 2,
        min_sentence_coverage: float = 0.5,
        null_exclusion_s: float = 30.0,
        min_null_windows: int = 3,
        align_tolerance_s: float = 3.0,
        nearby_reach_s: float = 10.0,
        verbatim_measure: str = "screen_coverage",
        semantic_measure: str = "best_matching_sentence",
    ) -> None:
        total = weight_verbatim + weight_semantic
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")
        self.embedder = embedder
        self.weight_verbatim = weight_verbatim
        self.weight_semantic = weight_semantic
        self.min_onscreen_words = min_onscreen_words
        self.min_narration_words = min_narration_words
        self.norm = norm or NormalisationConfig()
        self.embedding_model_name = embedding_model_name
        self.min_shared_content_words = min_shared_content_words
        self.min_sentence_coverage = min_sentence_coverage
        self.null_exclusion_s = null_exclusion_s
        self.min_null_windows = min_null_windows
        if verbatim_measure not in ("screen_coverage", "f1"):
            raise ValueError(f"unknown verbatim_measure {verbatim_measure!r}")
        if semantic_measure not in ("best_matching_sentence", "whole_window"):
            raise ValueError(f"unknown semantic_measure {semantic_measure!r}")
        self.verbatim_measure = verbatim_measure
        self.align_tolerance_s = align_tolerance_s
        self.nearby_reach_s = nearby_reach_s
        self._words: tuple = ()
        self._showings: dict[str, list[list[float]]] = {}
        self.semantic_measure = semantic_measure
        self._vec_cache: dict[str, Any] = {}
        # A static slide spans many windows, so the same (ocr, asr) text pair
        # recurs. Cache to avoid re-encoding identical strings.
        self._sem_cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_config(cls, cfg, embedder: Embedder) -> RedundancyScorer:
        r = "scoring.redundancy."
        return cls(
            embedder=embedder,
            weight_verbatim=float(cfg.get(r + "weights.verbatim")),
            weight_semantic=float(cfg.get(r + "weights.semantic")),
            min_onscreen_words=int(cfg.get(r + "min_onscreen_words")),
            min_narration_words=int(cfg.get(r + "min_narration_words", 1)),
            norm=NormalisationConfig.from_dict(cfg.get("text_normalisation", {})),
            embedding_model_name=str(cfg.get(r + "embedding_model")),
            min_shared_content_words=int(
                cfg.get(r + "chance_correction.min_shared_content_words", 2)
            ),
            min_sentence_coverage=float(
                cfg.get(r + "chance_correction.min_sentence_coverage", 0.5)
            ),
            null_exclusion_s=float(cfg.get(r + "chance_correction.null_exclusion_s", 30.0)),
            min_null_windows=int(cfg.get(r + "chance_correction.min_null_windows", 3)),
            align_tolerance_s=float(cfg.get("alignment.tolerance_s", 3.0)),
            nearby_reach_s=float(cfg.get(r + "nearby_reach_s", 10.0)),
            verbatim_measure=str(cfg.get(r + "verbatim_measure", "screen_coverage")),
            semantic_measure=str(cfg.get(r + "semantic_measure", "best_matching_sentence")),
        )

    def params(self) -> dict[str, Any]:
        return {
            "weights": {
                "verbatim": self.weight_verbatim,
                "semantic": self.weight_semantic,
            },
            "min_onscreen_words": self.min_onscreen_words,
            "min_narration_words": self.min_narration_words,
            "embedding_model": self.embedding_model_name,
            "verbatim_measure": self.verbatim_measure,
            "semantic_measure": self.semantic_measure,
            "align_tolerance_s": self.align_tolerance_s,
            "nearby_reach_s": self.nearby_reach_s,
            "text_normalisation": vars(self.norm),
            "chance_correction": {
                "min_shared_content_words": self.min_shared_content_words,
                "min_sentence_coverage": self.min_sentence_coverage,
                "null_exclusion_s": self.null_exclusion_s,
                "min_null_windows": self.min_null_windows,
            },
        }

    # ------------------------------------------------------------------ #

    def score(self, windows, extra_warnings=(), context=None):
        """Score every window against chance in its own video, then aggregate.

        Each window's on-screen text is compared with narration from *other*
        moments of the same video -- windows starting more than
        `null_exclusion_s` away. That tells us how much the text matches
        narration it is not being read alongside: the floor below which a
        match is just the video's subject matter, register or chance. A
        lecture about determinants mentions determinants throughout; a caption
        about them is only redundant where the narration says it *then*.

        `context` (21 Oct 2026): when it carries the transcript, each on-screen
        sentence is compared with the narration spoken from `align_tolerance_s`
        before it appears to `align_tolerance_s` after it disappears -- its
        own time on screen, not the window's. Without it (unit tests, windows
        built by hand) the window's narration pool is used.
        """
        self._words = tuple(context.transcript.words) if (
            context is not None and getattr(context, "transcript", None) is not None) else ()
        self._showings = self._video_showings(windows)
        scores = []
        for w in windows:
            elsewhere = [
                x.asr_text
                for x in windows
                if x.asr_text.strip() and abs(x.start - w.start) >= self.null_exclusion_s
            ]
            scores.append(self.score_window(w, elsewhere=elsewhere))
        return self.aggregate(tuple(scores), extra_warnings=extra_warnings)

    def score_window(self, window: Window, elsewhere: Sequence[str] = ()) -> WindowScore:
        # Only prose elements can constitute redundancy. Keywords, topic titles
        # and labels are excluded both from the qualification decision and from
        # the compared text, per the operative definition (see onscreen.py).
        ocr_for_scoring = window.ocr_sentence_text
        ocr_words = word_count(ocr_for_scoring)
        asr_words = word_count(window.asr_text)
        counts = {
            "ocr_sentence_word_count": float(ocr_words),
            "ocr_total_word_count": float(word_count(window.ocr_text)),
            "asr_word_count": float(asr_words),
            "n_sentence_elements": float(
                sum(1 for e in window.ocr_elements if e.get("kind") == "sentence")
            ),
            "n_label_elements": float(
                sum(1 for e in window.ocr_elements if e.get("kind") == "label")
            ),
        }

        reason = self._exclusion_reason(ocr_words, asr_words)
        if reason is not None:
            return WindowScore(
                window=window,
                score=None,
                qualifies=False,
                components=counts,
                exclusion_reason=reason,
            )

        # The sentence-level gate (05 Oct 2026). Redundancy is narration that
        # restates an on-screen SENTENCE. The narration must say more than
        # `min_sentence_coverage` of one sentence's content words -- at least
        # `min_shared_content_words` of them -- and more than narration
        # elsewhere in the video says of that same sentence. Words picked from
        # all over a screen do not add up to a restated sentence: test4 showed
        # a page of a paper and the narration "Irene Tobler noticed that
        # cockroaches sleep" took "Irene Tobler" from the author line, "sleep"
        # from the journal name and "cockroaches" from the abstract. It scored
        # 21.27 on words alone; no sentence on screen is restated (best: 12%).
        sentences = self._screen_sentences(window)
        spans_of = self._sentence_spans(window)
        raw_spans_of = self._sentence_spans(window, clip=False)
        has_null = len(elsewhere) >= self.min_null_windows
        restated = []
        # Every sentence tested, passed or not, so a reader can follow the
        # whole decision -- including why a window scored 0 (the web UI's
        # segment breakdown shows these step by step).
        checks = []
        for sentence in sentences:
            # An on-screen question is restated only by being asked aloud
            # (user's ruling, 05 Oct 2026). Answering it repeats its words --
            # "What is the new volume of the solution?" / "So that's the new
            # volume of the solution." -- but a question and its answer are
            # different things, and the answer is not redundancy. So a question
            # is compared only with narration spoken as a question.
            heard, heard_from, heard_to = self._narration_for(sentence, raw_spans_of, window)
            if is_question(sentence):
                said, said_elsewhere = question_said_forms(heard), question_said_forms
            else:
                said, said_elsewhere = said_forms(heard), said_forms
            words, shared = shared_content(sentence, said)
            if not words:
                continue
            coverage = len(shared) / len(words)
            chance = (
                statistics.median(len(shared_content(sentence, said_elsewhere(n))[1]) / len(words)
                                  for n in elsewhere)
                if has_null
                else 0.0
            )
            passed = (
                len(shared) >= self.min_shared_content_words
                and coverage > self.min_sentence_coverage
                and coverage > chance
            )
            spans = spans_of.get(sentence, [[window.start, window.end]])
            checks.append({
                "sentence": sentence,
                "narration": heard,
                # the stretch of narration compared (None: the window's pool)
                "narration_from": None if heard_from is None else round(heard_from, 2),
                "narration_to": None if heard_to is None else round(heard_to, 2),
                "spans": [[round(a, 2), round(b, 2)] for a, b in spans],
                "visible_s": round(sum(b - a for a, b in spans), 2),
                "is_question": is_question(sentence),
                "content_words": sorted(words),
                "shared_words": sorted(shared),
                "coverage": round(coverage, 3),
                "chance": round(chance, 3),
                "passed": passed,
            })
            if passed:
                restated.append({
                    "sentence": sentence,
                    "coverage": round(coverage, 3),
                    "chance": round(chance, 3),
                    "shared_words": sorted(shared),
                })
        evidence = {
            "restated_sentences": restated,
            "shared_content_words": sorted({w for r in restated for w in r["shared_words"]}),
            "candidate_sentences": len(sentences),
            "sentence_checks": checks,
            "null_windows": len(elsewhere) if has_null else 0,
        }

        if not restated:
            return WindowScore(
                window=window,
                score=0.0,
                qualifies=True,
                components={**counts, "verbatim": 0.0, "semantic": 0.0},
                exclusion_reason=None,
                evidence={**evidence, "gate": "failed"},
            )

        # How redundant (21 Oct 2026, user's decision): every on-screen
        # sentence is scored on its own -- 0 if the narration does not restate
        # it -- and the window score is the average, over the half-seconds in
        # which a sentence is on screen, of the length-weighted score of the
        # sentences visible at that moment. A caption counts for exactly the
        # seconds it is visible, so one that crosses a window boundary is
        # shared between the windows, not counted in full twice (user,
        # animation_london_subtitled: "The big wheel …" is on screen 7.0-11.5 s
        # and had counted fully in both 0:00-0:10 and 0:10-0:20).
        restated_set = {r["sentence"] for r in restated}
        per_sentence = {}
        for chk in checks:
            sent = chk["sentence"]
            if sent not in restated_set:
                per_sentence[sent] = {"verbatim": 0.0, "semantic": 0.0}
                continue
            cw = self._content_text(sent)
            heard = chk["narration"]
            v_raw = self._verbatim_sentences([sent], cw, heard)
            s_raw = self._meaning([sent], sent, heard)
            v_ch = statistics.median(self._verbatim_sentences([sent], cw, n)
                                     for n in elsewhere) if has_null else 0.0
            s_ch = statistics.median(self._meaning([sent], sent, n)
                                     for n in elsewhere) if has_null else 0.0
            per_sentence[sent] = {
                "verbatim_raw": round(v_raw, 4), "verbatim_chance": round(v_ch, 4),
                "verbatim": round(_above_chance(v_raw, v_ch), 4),
                "semantic_raw": round(s_raw, 4), "semantic_chance": round(s_ch, 4),
                "semantic": round(_above_chance(s_raw, s_ch), 4),
            }
        for chk in checks:
            ps = per_sentence[chk["sentence"]]
            chk.update(ps)
            chk["score"] = round(100.0 * (self.weight_verbatim * ps["verbatim"]
                                          + self.weight_semantic * ps["semantic"]), 2)

        verbatim, semantic, on_screen = self._time_weighted(window, checks)
        combined = self.weight_verbatim * verbatim + self.weight_semantic * semantic
        # Kept for audit and comparison: the measure used until 15 Oct 2026.
        all_cw = self._content_text(" ".join(r["sentence"] for r in restated))
        counts["verbatim_f1_raw"] = round(
            rouge_l_f1(all_cw, self._content_text(window.asr_text), self.norm), 4)

        return WindowScore(
            window=window,
            score=round(100.0 * combined, 2),
            qualifies=True,
            components={
                **counts,
                "verbatim": round(verbatim, 4),
                "semantic": round(semantic, 4),
                "screen_time_share": round(on_screen, 4),
            },
            exclusion_reason=None,
            evidence={**evidence, "gate": "passed", "time_step_s": TIME_STEP_S},
        )

    def _time_weighted(self, window: Window, checks: list[dict]) -> tuple[float, float, float]:
        """(verbatim, semantic, share of time with a sentence on screen),
        averaged over the window's half-seconds; see score_window."""
        n = max(1, round((window.end - window.start) / TIME_STEP_S))
        step = (window.end - window.start) / n
        v_sum = s_sum = shown = 0.0
        share = [0.0] * len(checks)
        for k in range(n):
            t = window.start + (k + 0.5) * step
            visible = [c for c in checks
                       if any(a <= t < b for a, b in c["spans"])]
            if not visible:
                continue
            shown += 1
            weights = [max(1, len(c["content_words"])) for c in visible]
            total = sum(weights)
            for wt, c in zip(weights, visible, strict=True):
                share[next(i for i, x in enumerate(checks) if x is c)] += wt / total
            v_sum += sum(w * c.get("verbatim", 0.0) for w, c in zip(weights, visible, strict=True)) / total
            s_sum += sum(w * c.get("semantic", 0.0) for w, c in zip(weights, visible, strict=True)) / total
        # Each sentence's exact share of the window's average, so the report
        # can show the window score as a weighted sum of sentence scores.
        for c, x in zip(checks, share, strict=True):
            c["time_weight"] = round(x / shown, 4) if shown else 0.0
        if not shown:
            return 0.0, 0.0, 0.0
        # Averaged over the moments a sentence is on screen. Seconds with no
        # recognised sentence are left out of the window's average (21 Oct
        # 2026): counting them as 0 turned every gap in sentence recognition --
        # a subtitle fragment not stitched, a logo glued onto a caption -- into
        # "nothing redundant", and contradicted the user's confirmed
        # expectations for Test_28 (R4-R6, R10b). Windows with no sentence at
        # all still count 0 in the segment and video averages.
        return v_sum / shown, s_sum / shown, shown / n

    def _narration_for(self, sentence: str, raw_spans: dict,
                       window: Window) -> tuple[str, float | None, float | None]:
        """The narration a sentence on screen in this window is compared with.

        The NEARBY rule (user's decision, 22 Oct 2026): narration spoken while
        the sentence is on screen -- its whole showing, across window edges --
        but no further than `nearby_reach_s` beyond the window on either side,
        widened by `align_tolerance_s`. A slide whose three sentences are read
        one per 10 s counts as read aloud throughout, as a rater watching it
        would judge; a slide read once and then left up stops counting one
        window later (test_1). Rejected alternatives, both measured on the
        corpus on 21 Oct: only the part on screen in this window (+-3 s), which
        scored slide decks read gradually as ~30 (Test_18 29 -> 11.5); and the
        whole showing uncapped, which compared a slide up for the whole video
        with the whole video's narration (test_1 4 -> 70).
        """
        spans = raw_spans.get(sentence)
        if not self._words or not spans:
            return window.asr_text, None, None
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        for a, b in self._showings.get(_showing_key(sentence), ()):
            if a <= hi + _SAME_SHOWING_GAP_S and b >= lo - _SAME_SHOWING_GAP_S:
                lo, hi = min(lo, a), max(hi, b)
        lo = max(lo, window.start - self.nearby_reach_s) - self.align_tolerance_s
        hi = min(hi, window.end + self.nearby_reach_s) + self.align_tolerance_s
        return " ".join(w.text.strip() for w in self._words if lo <= w.midpoint < hi), lo, hi

    def _video_showings(self, windows: Sequence[Window]) -> dict[str, list[list[float]]]:
        """Every continuous showing of each on-screen sentence in the video
        (gaps up to `_SAME_SHOWING_GAP_S` bridged), keyed by its normalised text."""
        times: dict[str, list] = {}
        for w in windows:
            for sentence, spans in self._sentence_spans(w, clip=False).items():
                times.setdefault(_showing_key(sentence), []).extend(spans)
        out = {}
        for key, spans in times.items():
            runs: list[list[float]] = []
            for a, b in sorted((float(a), float(b)) for a, b in spans):
                if runs and a <= runs[-1][1] + _SAME_SHOWING_GAP_S:
                    runs[-1][1] = max(runs[-1][1], b)
                else:
                    runs.append([a, b])
            out[key] = runs
        return out

    def _sentence_spans(self, window: Window, clip: bool = True) -> dict[str, list[list[float]]]:
        """When each on-screen sentence is visible, from its element's
        recorded spans -- clipped to the window unless `clip` is False.
        Elements without spans count for the whole window."""
        out: dict[str, list[list[float]]] = {}
        for e in window.ocr_elements:
            if e.get("kind") != "sentence":
                continue
            raw = e.get("spans") or [[window.start, window.end]]
            if not clip:
                for sent in self._sentences_of_element(e):
                    out.setdefault(sent, []).extend([list(x) for x in raw])
                continue
            spans = [[max(a, window.start), min(b, window.end)] for a, b in raw
                     if b > window.start and a < window.end]
            if not spans:
                continue
            for sent in self._sentences_of_element(e):
                out.setdefault(sent, []).extend(spans)
        return out

    def _screen_sentences(self, window: Window) -> tuple[str, ...]:
        """Every on-screen sentence in the window, one per bullet/sentence."""
        out: list[str] = []
        for e in window.ocr_elements:
            if e.get("kind") == "sentence":
                out.extend(self._sentences_of_element(e))
        if not out and window.ocr_sentence_text.strip() and not window.ocr_elements:
            out.extend(onscreen_sentences(window.ocr_sentence_text,
                                          self.min_shared_content_words))
        return tuple(dict.fromkeys(out))

    def _sentences_of_element(self, e: dict) -> list[str]:
        if e.get("reason") == "stitched_across_screens":
            # Already classified as a whole sentence; its text here is the
            # part visible in this window, which may be a fragment and must
            # not be re-classified (21 Oct 2026). Still split at sentence ends,
            # since a window can show the end of one caption sentence and the
            # start of the next.
            return [piece.strip() for piece in _SENTENCE_SPLIT.split(" ".join(e["text"].split()))
                    if len(content_lemmas(piece)) >= self.min_shared_content_words]
        return list(onscreen_sentences(e["text"], self.min_shared_content_words))

    def _content_words(self, text: str) -> frozenset[str]:
        # Words of 3+ letters, and numbers of 2+ characters ("25", "300",
        # "0.25"): in a worked problem, the numbers are what gets read aloud.
        return frozenset(
            w
            for w in self._content_text(text).split()
            if len(w) >= 3 or (len(w) >= 2 and any(c.isdigit() for c in w))
        )

    def _content_text(self, text: str) -> str:
        """Normalised text with function words removed.

        "is", "the", "your", "so" are shared by any two English passages; on
        Test_11 they gave a Seinfeld joke and an unrelated sentence of
        narration a verbatim overlap of 0.12. Duplication is about content.
        """
        return " ".join(
            w for w in normalise_for_verbatim(text, self.norm).split() if w not in STOP_WORDS
        )

    def _exclusion_reason(self, ocr_words: int, asr_words: int) -> str | None:
        """Why this window carries no measurable redundancy.

        `ocr_words` counts only prose. A window whose screen is full of labels
        therefore reports NO_ONSCREEN_TEXT rather than the signaling exception:
        there was text, but none of it was the kind that can be redundant. The
        element breakdown in `components` distinguishes the two cases.
        """
        if ocr_words == 0:
            return EXCL_NO_ONSCREEN_TEXT
        if ocr_words < self.min_onscreen_words:
            return EXCL_SIGNALING_EXCEPTION
        if asr_words < self.min_narration_words:
            return EXCL_NO_NARRATION
        return None

    def _unmeasurable_reason(self, window_scores) -> str | None:
        """Distinguish "no redundancy present" (score 0) from "cannot assess" (N/A).

        Redundancy requires narration and on-screen text to coexist. If narration
        is absent everywhere, the principle cannot be evaluated at all and the
        honest answer is N/A -- a silent track is far more often a broken audio
        extraction than a deliberate pedagogical choice, and scoring it 0 would
        dress a failure up as an ideal video.

        Every other way of failing to qualify (no on-screen text, or only short
        signaling labels) means we did assess the video and found nothing that
        could constitute redundancy. That is the floor of the scale: 0.
        """
        base = super()._unmeasurable_reason(window_scores)
        if base is not None:
            return base

        if all(ws.exclusion_reason == EXCL_NO_NARRATION for ws in window_scores):
            return (
                f"no narration detected in any of {len(window_scores)} windows — "
                "redundancy is undefined without speech to compare against. Check "
                "the audio track before treating this as a property of the video."
            )
        return None

    def _verbatim_sentences(self, sentences: Sequence[str], joined_cw: str,
                            narration: str) -> float:
        """Screen coverage measured sentence by sentence, weighted by length
        (21 Oct 2026). Joining the restated sentences first made the score depend
        on the ORDER they sit on screen: LCS needs the same order, and four
        sentences all read aloud scored 0.65 because the screen listed them in a
        different order from the speech (Test_28, 0:20). F1 keeps the joined form."""
        if self.verbatim_measure == "f1":
            return self._verbatim(joined_cw, narration)
        total = weight = 0.0
        for sentence in sentences:
            cw = self._content_text(sentence)
            n = len(cw.split())
            if n:
                total += n * self._verbatim(cw, narration)
                weight += n
        return total / weight if weight else 0.0

    def _verbatim(self, screen_cw: str, narration: str) -> float:
        """Word match of the restated on-screen content words against narration."""
        other = self._content_text(narration)
        if self.verbatim_measure == "f1":
            return rouge_l_f1(screen_cw, other, self.norm)
        return rouge_l_screen_coverage(screen_cw, other, self.norm)

    def _meaning(self, sentences: Sequence[str], joined: str, narration: str) -> float:
        """Meaning match of the restated on-screen sentences against narration.

        best_matching_sentence (15 Oct 2026, user decision): each on-screen
        sentence is compared with the narration sentence (or consecutive pair)
        that matches it best, and the results are averaged weighted by content
        words -- so speech around it that the screen does not show does not
        dilute the comparison. whole_window is the measure used until then.
        """
        if self.semantic_measure == "whole_window":
            return clamp_similarity(self._semantic(joined, narration))
        units = narration_units(narration)
        if not units:
            return 0.0
        words = narration.split()
        total = weight = 0.0
        for s in sentences:
            w = max(1, len(self._content_words(s)))
            # Spoken sentences (and pairs), plus every stretch of narration as
            # long as the on-screen text (21 Oct 2026): a caption fragment on
            # screen ("on fire prevention and emergency preparation.") is
            # restated by PART of a spoken sentence, and comparing it with the
            # whole sentence understated it (0.84 for words read verbatim,
            # Test_28 1:50). The chance level is computed the same way.
            n = len(s.split())
            spans = [" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))]
            best = max(self._similarities(s, list(dict.fromkeys(units + spans))))
            total += w * clamp_similarity(best)
            weight += w
        return total / weight if weight else 0.0

    def _similarities(self, a: str, others: Sequence[str]) -> list[float]:
        """Cosine of `a` with each of `others`, encoding uncached texts in one batch."""
        encode = getattr(self.embedder, "encode", None)
        if encode is None:
            return [self._semantic(a, b) for b in others]
        todo = [t for t in dict.fromkeys([a, *others]) if t not in self._vec_cache]
        if todo:
            for t, v in zip(todo, encode(todo), strict=True):
                self._vec_cache[t] = v
        va = self._vec_cache[a]
        return [float(va @ self._vec_cache[b]) for b in others]

    def _similarity(self, a: str, b: str) -> float:
        """Cosine of two texts; cached per text when the embedder can encode."""
        encode = getattr(self.embedder, "encode", None)
        if encode is None:
            return self._semantic(a, b)
        for t in (a, b):
            if t not in self._vec_cache:
                self._vec_cache[t] = encode([t])[0]
        return float(self._vec_cache[a] @ self._vec_cache[b])

    def _semantic(self, ocr_text: str, asr_text: str) -> float:
        key = (ocr_text, asr_text)
        if key not in self._sem_cache:
            self._sem_cache[key] = self.embedder.encode_pair(ocr_text, asr_text)
        return self._sem_cache[key]
