"""Statistics for validation, chosen for a small sample (tens of videos).

  * Spearman's rho is the target (CLAUDE.md: >= 0.7). It uses ranks only, so the
    professor's scale (1-5, 0-10, 0-100) does not matter; ties (many videos
    score 0) get average ranks. Pearson's r is reported beside it.
  * Uncertainty is a 95% percentile bootstrap interval (10,000 resamples, fixed
    seed: reproducible). At segment level the resampling unit is the VIDEO --
    segments of one video are not independent, and resampling them separately
    would make the interval falsely narrow.
  * The tune / held-out split is by video, seeded, and recorded. A video's
    segments never fall on both sides.
  * Variant choice happens on the tune split only (CLAUDE.md §5). Nested
    leave-one-video-out is reported beside it: with few videos a single split
    is itself a lucky or unlucky draw.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(sps.spearmanr(x, y).statistic)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(sps.pearsonr(x, y).statistic)


def spearman_p(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(sps.spearmanr(x, y).pvalue)


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[str] | None = None,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    stat=spearman,
) -> tuple[float, float, int]:
    """95% percentile interval for `stat`, resampling `groups` (default: rows).
    Returns (low, high, n_valid_resamples); resamples where the statistic is
    undefined (e.g. all-equal scores) are dropped and counted."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    groups = np.asarray(groups if groups is not None else np.arange(len(x)))
    units = np.unique(groups)
    index = {u: np.flatnonzero(groups == u) for u in units}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        pick = rng.choice(units, size=len(units), replace=True)
        rows = np.concatenate([index[u] for u in pick])
        v = stat(x[rows], y[rows])
        if not math.isnan(v):
            values.append(v)
    if len(values) < n_resamples * 0.5:
        return math.nan, math.nan, len(values)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), len(values)


def split_videos(video_ids: Sequence[str], tune_fraction: float, seed: int):
    """Seeded split of videos into (tune, held_out), both sorted."""
    ids = sorted(set(video_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_tune = max(1, min(len(ids) - 1, round(len(ids) * tune_fraction)))
    return sorted(ids[:n_tune]), sorted(ids[n_tune:])


def choose_variant(frame, variants: Sequence[str], manual: str = "manual") -> tuple[str, dict]:
    """The variant with the highest Spearman on `frame`; ties and undefined
    values go to the earlier (primary-first) variant."""
    rhos = {v: spearman(frame[v], frame[manual]) for v in variants}
    best = variants[0]
    for v in variants[1:]:
        if not math.isnan(rhos[v]) and (math.isnan(rhos[best]) or rhos[v] > rhos[best]):
            best = v
    return best, rhos


def nested_leave_one_video_out(frame, variants: Sequence[str], video_col: str = "video_id"):
    """For each video: choose the variant on all other videos, then take that
    variant's score(s) for the left-out video. Returns (out_of_fold scores
    aligned with `frame`, {video: chosen variant})."""
    oof = np.full(len(frame), np.nan)
    chosen = {}
    vids = frame[video_col].to_numpy()
    for vid in sorted(set(vids)):
        rest = frame[vids != vid]
        best, _ = choose_variant(rest, variants)
        chosen[vid] = best
        mask = vids == vid
        oof[mask] = frame.loc[mask, best].to_numpy(dtype=float)
    return oof, chosen


def inter_rater(frame, raters: Sequence[str]) -> list[dict]:
    """Spearman between every pair of raters, on rows both scored."""
    out = []
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            both = frame[[a, b]].dropna()
            out.append({"raters": [a, b], "n": len(both),
                        "spearman": spearman(both[a], both[b])})
    return out
