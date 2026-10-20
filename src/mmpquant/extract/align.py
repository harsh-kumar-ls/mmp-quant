"""Alignment layer: turn a transcript + an OCR track into scored-ready `Window`s.

This module is where the ±3 s tolerance is actually defined, so the definition
is spelled out here rather than left implicit.

Why a tolerance is needed
-------------------------
A narrator says "as you can see here, mitochondria produce ATP" and the slide
appears half a second later, or stays up for three seconds after they have moved
on. With hard 10 s window edges, that offset splits a genuinely redundant
pairing across two windows and both score low. The tolerance absorbs it.

Which side gets widened (`alignment.expand`)
--------------------------------------------
Widening *both* sides is the obvious reading of "±3 s" and it is the wrong
choice: each window would then pull text from a 22 s span (10 + 6 + 6), so
adjacent windows overlap heavily, the same narration is counted repeatedly
against different slides, and overlap is inflated across the board.

The default is `asr`: the on-screen text defines the window's extent, because
the slide is what the learner is looking at, and the narration pool is widened
to ±3 s around it. This keeps the OCR side partitioned (no double counting of
slides) while still absorbing narration lead/lag. `both` is available for
sensitivity analysis and should be reported as such if used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ..onscreen import (
    KIND_LABEL,
    KIND_SENTENCE,
    TextElement,
    collapse_near_duplicates,
    group_into_elements,
    group_lines,
    prose_text,
)
from ..types import OcrBlock, OcrLine, Transcript, Window
from ..wordsplit import split_run_together
from .captions import StitchedSentence, caption_stream, stitch_sentences

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowingParams:
    size_s: float = 10.0
    tolerance_s: float = 3.0
    expand: str = "asr"  # asr | ocr | both
    keep_partial_final_window: bool = True

    def __post_init__(self) -> None:
        if self.size_s <= 0:
            raise ValueError("size_s must be > 0")
        if self.tolerance_s < 0:
            raise ValueError("tolerance_s must be >= 0")
        if self.expand not in ("asr", "ocr", "both"):
            raise ValueError(f"expand must be asr|ocr|both, got {self.expand!r}")

    @property
    def asr_tolerance(self) -> float:
        return self.tolerance_s if self.expand in ("asr", "both") else 0.0

    @property
    def ocr_tolerance(self) -> float:
        return self.tolerance_s if self.expand in ("ocr", "both") else 0.0


@dataclass(frozen=True)
class OnscreenParams:
    """Everything that decides how OCR lines become sentence or label elements.

    Before defect 10.3 (04 Oct 2026) only `min_prose_words` was passed down
    from config; `line_gap_ratio`, `x_align_tol_px` and `dedup_similarity` were
    documented in config.yaml as tunable but never read -- the code used
    hard-coded defaults that happened to equal the config values, so editing
    the config silently changed nothing. They now travel together.
    """

    min_prose_words: int = 5
    line_gap_ratio: float = 1.6
    x_align_tol_px: float = 24.0
    dedup_similarity: float = 0.88
    split_run_together_words: bool = True
    # Captions running one sentence across screens (15 Oct 2026, captions.py)
    caption_stitching: bool = True
    caption_max_gap_s: float = 1.0
    caption_max_pieces: int = 4
    # Highlighted key lines are signaling, not redundancy (20 Oct 2026)
    highlighted_key_line: bool = True
    highlight_max_words: int = 12
    highlight_min_size_ratio: float = 1.5


def window_bounds(duration: float, params: WindowingParams) -> list[tuple[float, float]]:
    """Contiguous, non-overlapping [start, end) windows covering [0, duration)."""
    if duration <= 0:
        return []
    # A tail shorter than this is discarded rather than kept as a window.
    # Durations are reported to microsecond precision, so a video of "10.0001 s"
    # otherwise yields a second window of 0.0001 s: it can hold no narration and
    # no slide, yet it enters the duration-weighted mean with a weight of
    # essentially zero, and would divide by ~0 if it were the only one to qualify.
    MIN_WINDOW_S = 0.1

    bounds: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(start + params.size_s, duration)
        is_partial = (end - start) < params.size_s - 1e-9
        if is_partial and not params.keep_partial_final_window:
            break
        if (end - start) < MIN_WINDOW_S:
            break
        bounds.append((start, end))
        start = end
    return bounds


def build_windows(
    transcript: Transcript,
    ocr_blocks: tuple[OcrBlock, ...],
    duration: float,
    params: WindowingParams,
    min_prose_words: int = 5,
    onscreen: OnscreenParams | None = None,
) -> list[Window]:
    """Assign narration and on-screen text to each window.

    Assignment rules:
      * A transcript word joins a window if its *midpoint* falls in
        [start - asr_tol, end + asr_tol). Midpoint rather than overlap so a long
        word straddling a boundary lands in exactly one window's core span
        rather than being duplicated by the untolerated rule.
      * An OCR block joins a window if its [start, end) span *intersects*
        [start - ocr_tol, end + ocr_tol). Intersection rather than midpoint
        because a block can be much longer than a window — a slide held for 60 s
        is on screen during all six of its windows and must count in each.
    """
    if onscreen is None:
        onscreen = OnscreenParams(min_prose_words=min_prose_words)
    words = transcript.words
    bounds = window_bounds(duration, params)
    windows: list[Window] = []
    stitched = _stitched_sentences(ocr_blocks, onscreen) if onscreen.caption_stitching else []
    usual_height = _median_line_height(ocr_blocks)

    for index, (start, end) in enumerate(bounds):
        asr_lo, asr_hi = start - params.asr_tolerance, end + params.asr_tolerance
        ocr_lo, ocr_hi = start - params.ocr_tolerance, end + params.ocr_tolerance

        window_words = [w for w in words if asr_lo <= w.midpoint < asr_hi]
        asr_text = " ".join(w.text for w in window_words)

        block_indices = [
            i
            for i, b in enumerate(ocr_blocks)
            if b.start < ocr_hi and b.end > ocr_lo
        ]
        ocr_text = _merge_block_text(ocr_blocks, block_indices)
        elements = _elements_for_window(ocr_blocks, block_indices, onscreen)
        elements = _with_stitched(
            elements, [s for s in stitched if s.start < ocr_hi and s.end > ocr_lo]
        )
        if onscreen.highlighted_key_line:
            elements = _mark_highlighted(elements, usual_height, onscreen)
        prose = prose_text(elements)

        windows.append(
            Window(
                index=index,
                start=start,
                end=end,
                asr_text=asr_text,
                ocr_text=ocr_text,
                ocr_sentence_text=prose,
                ocr_elements=tuple(e.to_dict() for e in elements),
                asr_pool_start=asr_lo,
                asr_pool_end=asr_hi,
                ocr_block_indices=tuple(block_indices),
            )
        )

    return windows


def _block_lines(block: OcrBlock, onscreen: OnscreenParams) -> tuple[OcrLine, ...]:
    if not onscreen.split_run_together_words:
        return block.lines
    # OCR sometimes drops the spaces between correctly-read words; restore
    # them before classification and comparison (defect 10.4, wordsplit.py).
    return tuple(
        OcrLine(split_run_together(line.text), line.confidence, line.bbox)
        for line in block.lines
    )


def _stitched_sentences(
    blocks: tuple[OcrBlock, ...], onscreen: OnscreenParams
) -> list[StitchedSentence]:
    """Sentences formed from captions that run across screens (captions.py)."""
    texts = [
        [(" ".join(line.text for line in group), _union_box(group))
         for group in group_lines(_block_lines(b, onscreen),
                                  line_gap_ratio=onscreen.line_gap_ratio,
                                  x_align_tol_px=onscreen.x_align_tol_px)]
        for b in blocks
    ]
    pieces = caption_stream(texts, [(b.start, b.end) for b in blocks],
                            similarity=onscreen.dedup_similarity,
                            max_gap_s=onscreen.caption_max_gap_s)
    return stitch_sentences(pieces, max_gap_s=onscreen.caption_max_gap_s,
                            max_pieces=onscreen.caption_max_pieces)


def _median_line_height(blocks: tuple[OcrBlock, ...]) -> float | None:
    """The video's usual text size: median height of every OCR line with a box."""
    import statistics

    heights = [ln.bbox[3] - ln.bbox[1] for b in blocks for ln in b.lines
               if ln.bbox and ln.bbox[3] > ln.bbox[1]]
    return statistics.median(heights) if heights else None


def _mark_highlighted(elements: list, usual_height: float | None,
                      onscreen: OnscreenParams) -> list:
    """Highlighted key lines become labels (user's ruling, 20 Oct 2026).

    A highlighted point -- one short statement set apart in large type, like
    Test_13's "All areas get scaled by det(A)" under the animation, or Test_25's
    centred heading "THE ARTS ARE VICTIMS" -- is signaling, not redundancy, even
    when the narration says it (cf. Mayer & Johnson 2008: brief on-screen key
    phrases beside narration help rather than hurt). Measured as: a sentence
    element that is ONE line, at most `highlight_max_words` words, at least
    `highlight_min_size_ratio` x the video's usual (median) line height, and not
    a stitched caption. The user ruled the same day that whiteboard bubble
    phrases and short slide statements are NOT highlighted points; none of
    those is a single line in large type, so the rule leaves them alone. On the
    20 Oct corpus any ratio from 1.1 to 1.9 selects the same two lines.
    """
    if not usual_height:
        return elements
    out = []
    for e in elements:
        n_lines = len(e.lines)
        if (e.kind == KIND_SENTENCE and e.bbox and n_lines == 1
                and e.evidence.get("reason") != "stitched_across_screens"
                and e.word_count <= onscreen.highlight_max_words):
            ratio = (e.bbox[3] - e.bbox[1]) / usual_height
            if ratio >= onscreen.highlight_min_size_ratio:
                e = replace(e, kind=KIND_LABEL, evidence={
                    **e.evidence, "reason": "highlighted_key_line",
                    "size_ratio": round(ratio, 2)})
        out.append(e)
    return out


def _union_box(lines) -> tuple[float, float, float, float] | None:
    boxes = [line.bbox for line in lines if line.bbox]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _key(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _with_stitched(elements: list, stitched: list[StitchedSentence]) -> list:
    """Add stitched sentences as sentence elements, dropping any per-screen
    sentence that is only a part of one (keep the longer, as dedupe does)."""
    if not stitched:
        return elements
    keys = [_key(s.text) for s in stitched]
    kept = [
        e for e in elements
        if not (e.kind == KIND_SENTENCE and any(_key(e.text) in k for k in keys))
    ]
    for s in stitched:
        kept.append(TextElement(
            lines=(OcrLine(s.text, 1.0, None),),
            kind=KIND_SENTENCE,
            bbox=None,
            evidence={"reason": "stitched_across_screens", "pieces": list(s.pieces),
                      "start": s.start, "end": s.end},
        ))
    return kept


def _elements_for_window(
    blocks: tuple[OcrBlock, ...], indices: list[int], onscreen: OnscreenParams
) -> list:
    """Reconstruct the window's on-screen text elements from its OCR blocks.

    Each block is grouped into elements on its own -- only lines that were on
    screen *at the same moment* can form one element -- and the elements from
    every block in the window are then de-duplicated together, so text held
    across several blocks, or captured at several stages of an animated render,
    is counted once.

    Until 04 Oct 2026 the lines of all blocks were pooled *before* grouping. A
    10 s window often spans a slide change, and text from two different moments
    that happened to sit in the same place on screen was then grouped as one
    element. On Test_11 the title "Gaussian Elimination" (shown first) and the
    label "Signed Area = det(A)y" (shown seconds later, just below) merged into
    "Gaussian Elimination Signed Area = det(A)y" -- which parses as a sentence,
    "Gaussian Elimination *signed* Area" -- and scored as redundancy. Neither
    was a sentence, and they were never on screen together.
    """
    elements = [
        element
        for i in indices
        for element in group_into_elements(
            _block_lines(blocks[i], onscreen),
            min_prose_words=onscreen.min_prose_words,
            line_gap_ratio=onscreen.line_gap_ratio,
            x_align_tol_px=onscreen.x_align_tol_px,
        )
    ]
    return collapse_near_duplicates(
        elements, similarity_threshold=onscreen.dedup_similarity
    )


def _merge_block_text(blocks: tuple[OcrBlock, ...], indices: list[int]) -> str:
    """Concatenate the text of several OCR blocks, dropping repeated lines.

    Two consecutive blocks are usually the same slide re-detected after a minor
    visual change (a cursor moved, a bullet appeared). Emitting their shared
    lines twice would double the on-screen word count and push a short signaling
    label over `min_onscreen_words`, wrongly turning it into a redundancy hit.
    Exact-duplicate lines are therefore kept once, in first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for i in indices:
        for line in blocks[i].lines:
            key = line.text.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return "\n".join(out)


def blocks_from_groups(
    frame_times: list[float],
    groups: list,
    lines_per_group: list[tuple],
    video_duration: float,
) -> tuple[OcrBlock, ...]:
    """Turn frame groups + their OCR results into time-extended `OcrBlock`s.

    A group's span runs from its first frame's timestamp to the next group's
    first frame (or the video end for the last group), so the blocks tile the
    whole timeline with no gaps — a window can never silently miss on-screen
    text that was present.
    """
    blocks: list[OcrBlock] = []
    for i, group in enumerate(groups):
        start = frame_times[group.first_index]
        end = (
            frame_times[groups[i + 1].first_index]
            if i + 1 < len(groups)
            else max(video_duration, frame_times[group.last_index])
        )
        blocks.append(
            OcrBlock(
                start=start,
                end=end,
                lines=tuple(lines_per_group[i]),
                source_frame_time=frame_times[group.representative_index],
                n_frames_collapsed=group.n_frames,
            )
        )
    return tuple(blocks)
