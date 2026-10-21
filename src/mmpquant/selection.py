"""Choosing the 10 videos the professor rates (rule fixed 20 Oct 2026, before
the new candidates were scored -- see PROGRESS.md).

  1. Eligible: every scored video <= max_duration_s (3:20).
  2. Redundancy targets 0, 10, ..., 90: one video per target, minimising the
     total |score - target| (optimal one-to-one assignment).
  3. Constraints, met only by swaps that raise a slot's cost by <= max_swap_cost:
     at least one video of each style; at least `min_music` with background
     music (coherence > music_cut) and at least `min_no_music` without.
  4. Ties: shorter video, then alphabetical.
  AMENDMENT (21 Oct 2026, user's decision, made AFTER seeing the round-2
  scores -- disclose in the thesis): among exact ties on redundancy distance,
  prefer a video that adds a coherence level not yet among the ten (none <= 10,
  low 10-30, mid 30-70, high > 70); only then the shorter video. It changes
  the sampling of coherence only, never the redundancy coverage.

The selection table carries the system's scores and is for the student only;
the professor receives just the videos and a blank workbook.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

STYLES = ("slides", "whiteboard", "pen", "animation")
TARGETS = tuple(range(0, 100, 10))


@dataclass(frozen=True)
class Candidate:
    video_id: str
    style: str
    duration_s: float
    redundancy: float
    coherence: float
    loudness_db: float | None

    @property
    def music(self) -> bool:
        return self.coherence > 10


@dataclass
class Selection:
    slots: list[tuple[int, Candidate]]  # (target, chosen)
    swaps: list[str] = field(default_factory=list)
    unmet: list[str] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(abs(c.redundancy - t) for t, c in self.slots)


def load_styles(path: Path) -> dict[str, str]:
    styles = {}
    if path.exists():
        rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
        for r in csv.DictReader(rows):
            styles[r["video_id"]] = r["style"].strip()
    return styles


def load_candidates(runs_dir: Path, videos: list[str], styles: dict[str, str],
                    max_duration_s: float) -> tuple[list[Candidate], dict[str, str]]:
    out, skipped = [], {}
    for v in videos:
        f = runs_dir / v / "result.json"
        if not f.exists():
            skipped[v] = "not scored yet"
            continue
        p = json.loads(f.read_text())
        if p["duration"] > max_duration_s:
            skipped[v] = f"too long ({p['duration']:.0f} s)"
            continue
        # Any labelled style is eligible (the rule's eligibility is duration
        # only); the four STYLES must each appear at least once. 'footage'
        # (filmed video with on-screen text) is eligible but is none of them.
        style = styles.get(v) or (v.split("_")[0] if v.split("_")[0] in STYLES else None)
        if not style:
            skipped[v] = "no style label"
            continue
        coh = p.get("coherence") or {}
        red = p["result"]["score"]
        if red is None:
            skipped[v] = "redundancy N/A"
            continue
        out.append(Candidate(v, style, p["duration"], red,
                             (coh.get("result") or {}).get("score") or 0.0,
                             (coh.get("music_loudness") or {}).get("median_db")))
    return out, skipped


def coherence_level(c: Candidate) -> str:
    x = c.coherence
    return "none" if x <= 10 else "low" if x <= 30 else "mid" if x <= 70 else "high"


def _spread_coherence(sel: Selection, cands: list[Candidate], min_music: int,
                      min_no_music: int) -> None:
    """The 21 Oct amendment: swap within exact redundancy ties to add a
    coherence level, never breaking a constraint."""
    for _ in range(len(sel.slots)):
        current = [c for _, c in sel.slots]
        levels = {coherence_level(c) for c in current}
        before = _deficits(current, min_music, min_no_music)
        best = None
        for i, (t, old) in enumerate(sel.slots):
            if sum(coherence_level(c) == coherence_level(old) for c in current) < 2:
                continue  # removing it would lose a level
            for new in cands:
                if new in current or coherence_level(new) in levels:
                    continue
                if abs(new.redundancy - t) > abs(old.redundancy - t) + 1e-9:
                    continue  # not a tie
                trial = current[:i] + [new] + current[i + 1:]
                if len(_deficits(trial, min_music, min_no_music)) > len(before):
                    continue
                key = (abs(new.redundancy - t), new.duration_s, new.video_id)
                if best is None or key < best[0]:
                    best = (key, i, new)
        if best is None:
            return
        _, i, new = best
        t, old = sel.slots[i]
        sel.slots[i] = (t, new)
        sel.swaps.append(f"target {t}: {old.video_id} -> {new.video_id} "
                         f"(adds coherence level '{coherence_level(new)}', tie, amendment 21 Oct)")


def _deficits(chosen: list[Candidate], min_music: int, min_no_music: int) -> list[str]:
    d = [f"style:{s}" for s in STYLES if not any(c.style == s for c in chosen)]
    if sum(c.music for c in chosen) < min_music:
        d.append("music")
    if sum(not c.music for c in chosen) < min_no_music:
        d.append("no_music")
    return d


def _fixes(c: Candidate, deficit: str) -> bool:
    if deficit.startswith("style:"):
        return c.style == deficit[6:]
    return c.music if deficit == "music" else not c.music


def select(cands: list[Candidate], *, targets=TARGETS, min_music: int = 3,
           min_no_music: int = 3, max_swap_cost: float = 5.0) -> Selection:
    if len(cands) < len(targets):
        raise ValueError(f"only {len(cands)} eligible videos for {len(targets)} targets")
    # deterministic order -> deterministic tie-breaking (shorter, then name)
    cands = sorted(cands, key=lambda c: (c.duration_s, c.video_id))
    cost = np.array([[abs(c.redundancy - t) for c in cands] for t in targets])
    rows, cols = linear_sum_assignment(cost)
    chosen = {targets[r]: cands[c] for r, c in zip(rows, cols, strict=True)}
    sel = Selection(slots=sorted(chosen.items()))

    for _ in range(20):
        current = [c for _, c in sel.slots]
        deficits = _deficits(current, min_music, min_no_music)
        if not deficits:
            break
        best = None
        for deficit in deficits:
            for i, (t, old) in enumerate(sel.slots):
                if _fixes(old, deficit):
                    continue  # replacing a video that already has it cannot help
                for new in cands:
                    if new in current or not _fixes(new, deficit):
                        continue
                    trial = current[:i] + [new] + current[i + 1:]
                    others = [d for d in _deficits(trial, min_music, min_no_music)
                              if d not in deficits]
                    added = abs(new.redundancy - t) - abs(old.redundancy - t)
                    if others or added > max_swap_cost:
                        continue
                    key = (added, new.duration_s, new.video_id)
                    if best is None or key < best[0]:
                        best = (key, i, new, deficit)
        if best is None:
            sel.unmet = deficits
            break
        (added, _, _), i, new, deficit = best
        t, old = sel.slots[i]
        sel.slots[i] = (t, new)
        sel.swaps.append(f"target {t}: {old.video_id} -> {new.video_id} "
                         f"(for {deficit}, +{added:.1f} points)")
    _spread_coherence(sel, cands, min_music, min_no_music)
    sel.unmet = _deficits([c for _, c in sel.slots], min_music, min_no_music)
    return sel


def write_table(sel: Selection, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "video_id", "style", "duration_s", "redundancy", "distance",
                    "coherence", "music_loudness_db"])
        for t, c in sel.slots:
            w.writerow([t, c.video_id, c.style, round(c.duration_s, 1), c.redundancy,
                        round(abs(c.redundancy - t), 2), c.coherence, c.loudness_db])
    return path
