"""Frame-diff deduplication.

A 10-minute lecture sampled at 2 fps is 1200 frames. If it shows 15 slides,
OCR'ing every frame does ~80x redundant work and produces 1200 near-identical
text blobs that then have to be re-deduplicated downstream. So we collapse runs
of visually near-identical frames into one block and OCR that block once.

The grouping core (`group_frames`) is pure numpy and takes signatures, not
files, so it is unit-testable with synthetic arrays and no video or OCR engine.

Anchor choice, which matters: each frame is compared against the *first* frame
of the current group, not against its immediate predecessor. Predecessor
comparison lets a slow cross-fade or a gradually-revealed bullet list drift
arbitrarily far from where the group started without any single step crossing
the threshold, silently merging two different slides. Anchoring bounds total
drift within a group.

Distance metric, which matters more
-----------------------------------
The default metric is the *fraction of pixels that changed appreciably*, not the
mean absolute difference. This was not an a-priori preference — mean absolute
difference was tried first and failed on the very first end-to-end run, merging
three visibly different slides into one block so that two of the three were
never OCR'd at all. Measured on that clip (1280x720 slides, black text on white,
compared at 320 px wide):

    frames of the same slide      mean-abs-diff 0.00000    changed-pixels 0.000%
    a genuinely different slide   mean-abs-diff 0.01003    changed-pixels 2.708%

Slide text covers only ~3 % of the pixels, so averaging a large per-pixel change
over a mostly-unchanged frame dilutes the signal by roughly 30x. The surviving
margin is so small that any mean-abs-diff threshold low enough to catch a
text-only slide change also sits inside the range of ordinary video compression
noise — there is no safe threshold, which makes it the wrong metric rather than a
badly-tuned one.

Counting *how many* pixels moved, instead of averaging *how far* the typical
pixel moved, is invariant to how much of the frame is ink. That is the property
we need, because lecture slides are mostly background.

`mean_abs_diff` is retained as a selectable method so the comparison above can be
reproduced, not because it is recommended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameGroup:
    """A run of visually near-identical sampled frames."""

    first_index: int
    last_index: int
    representative_index: int

    @property
    def n_frames(self) -> int:
        return self.last_index - self.first_index + 1


def frame_signature(image_path: Path, downscale_width: int = 320) -> np.ndarray:
    """Grayscale, downscaled float array in [0, 1] used for cheap comparison.

    Downscaling is deliberate, not just an optimisation: at full resolution,
    video compression noise and anti-aliasing on a static slide produce a
    non-trivial per-pixel difference, so no sane threshold separates "same
    slide" from "new slide". Downscaling averages that noise away while
    preserving the layout-level change a slide transition causes.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("L")
        if img.width > downscale_width:
            height = max(1, round(img.height * downscale_width / img.width))
            img = img.resize((downscale_width, height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


METHOD_CHANGED_PIXELS = "changed_pixel_ratio"
METHOD_MEAN_ABS_DIFF = "mean_abs_diff"


def frame_distance(
    a: np.ndarray,
    b: np.ndarray,
    method: str = METHOD_CHANGED_PIXELS,
    pixel_tolerance: float = 0.1,
) -> float:
    """Distance in [0, 1]. Shape mismatch means definitely different.

    `changed_pixel_ratio` (default): the fraction of pixels whose intensity moved
    by more than `pixel_tolerance`. The tolerance is what absorbs compression
    noise and anti-aliasing jitter, so the returned ratio reflects genuine
    layout change. See the module docstring for why this beats averaging.

    `mean_abs_diff`: retained for reproducing the comparison, not recommended.
    """
    if a.shape != b.shape:
        return 1.0
    if method == METHOD_MEAN_ABS_DIFF:
        return float(np.mean(np.abs(a - b)))
    if method == METHOD_CHANGED_PIXELS:
        return float(np.mean(np.abs(a - b) > pixel_tolerance))
    raise ValueError(
        f"unknown dedup method {method!r}; "
        f"expected one of {METHOD_CHANGED_PIXELS!r}, {METHOD_MEAN_ABS_DIFF!r}"
    )


def group_frames(
    signatures: list[np.ndarray],
    threshold: float = 0.005,
    method: str = METHOD_CHANGED_PIXELS,
    pixel_tolerance: float = 0.1,
) -> list[FrameGroup]:
    """Collapse consecutive near-identical signatures into groups.

    The default threshold of 0.005 means "more than 0.5 % of pixels moved". The
    measured signal for a real slide change is ~2.7 %, so this sits roughly 5x
    below the thing it must detect and well above the 0 % produced by repeated
    frames of a static slide.

    The representative is the group's middle frame: the first and last frames of
    a group are the ones most likely to be mid-transition (partially faded, or
    caught during a slide build), and OCR'ing a half-faded frame yields garbage
    or dropped lines.
    """
    if not signatures:
        return []

    groups: list[FrameGroup] = []
    anchor_index = 0
    anchor = signatures[0]

    for i in range(1, len(signatures)):
        if frame_distance(anchor, signatures[i], method, pixel_tolerance) > threshold:
            groups.append(_close_group(anchor_index, i - 1))
            anchor_index = i
            anchor = signatures[i]

    groups.append(_close_group(anchor_index, len(signatures) - 1))
    return groups


def _close_group(first: int, last: int) -> FrameGroup:
    return FrameGroup(
        first_index=first,
        last_index=last,
        representative_index=(first + last) // 2,
    )


def dedup_ratio(n_frames: int, n_groups: int) -> float:
    """Fraction of OCR work avoided. Logged so a dedup failure is visible.

    Near 0 means dedup did nothing — expect this on videos with a talking-head
    inset, an animated cursor, or continuous motion graphics, where every frame
    genuinely differs. In that case OCR cost approaches one call per sampled
    frame and `ocr.dedup.threshold` (or the sampling rate) needs revisiting for
    that video class. It is a performance signal, not a correctness one.
    """
    if n_frames <= 0:
        return 0.0
    return 1.0 - (n_groups / n_frames)
