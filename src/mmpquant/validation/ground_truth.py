"""Reading the professor's manual scores.

The format is not confirmed yet (CLAUDE.md §1), so the reader accepts the two
shapes this project has already anticipated, and refuses anything ambiguous
rather than guessing:

  * one CSV with a `video_id` column and one or more score columns
    (`manual_score`, `score`, or `<rater>_score`) -> one score per video;
  * the same with a `segment_index` column -> one score per 30 s segment;
  * a directory of CSVs -- e.g. the blank sheets written by
    `mmpquant annotation-sheet` (`data/annotations/*_sheet.csv`), once filled in.

Several score columns are several raters: the ground truth is their mean, and
their agreement with each other is reported (the honest ceiling for the
system). Blank cells are "not rated", never 0.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

KEY_VIDEO = "video_id"
KEY_SEGMENT = "segment_index"
_SCORE_COL = re.compile(r"^(manual_score|score|.+_score)$")
_DUMMY = re.compile(r"\b(dummy|synthetic|test data|not real)\b", re.IGNORECASE)


class GroundTruthError(ValueError):
    pass


@dataclass
class GroundTruth:
    level: str  # "video" | "segment"
    table: pd.DataFrame  # keys, rater columns, "manual" (mean of raters), "n_raters"
    raters: list[str]
    sources: list[str]
    sha256: str
    is_dummy: bool
    n_unrated_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return [KEY_VIDEO] if self.level == "video" else [KEY_VIDEO, KEY_SEGMENT]


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise GroundTruthError(f"no .csv files in {path}")
        return files
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def load_ground_truth(path: str | Path) -> GroundTruth:
    path = Path(path)
    files = _files(path)
    digest = hashlib.sha256()
    frames, is_dummy = [], _DUMMY.search(path.name) is not None
    for f in files:
        raw = f.read_bytes()
        digest.update(raw)
        text = raw.decode("utf-8-sig")
        comments = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
        if any(_DUMMY.search(c) for c in comments) or _DUMMY.search(f.name):
            is_dummy = True
        df = pd.read_csv(f, comment="#", dtype=str, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)

    columns = {tuple(df.columns) for df in frames}
    if len(columns) > 1:
        raise GroundTruthError(f"files disagree on their columns: {sorted(columns)}")
    df = pd.concat(frames, ignore_index=True)
    if KEY_VIDEO not in df.columns:
        raise GroundTruthError(f"no '{KEY_VIDEO}' column; found {list(df.columns)}")
    if "system_score" in df.columns:
        raise GroundTruthError(
            "a 'system_score' column means this is a system output "
            "(segment_scores.csv), not manual scores")

    level = "segment" if KEY_SEGMENT in df.columns else "video"
    keys = [KEY_VIDEO] if level == "video" else [KEY_VIDEO, KEY_SEGMENT]
    raters = [c for c in df.columns if _SCORE_COL.match(c) and c not in keys]
    if not raters:
        raise GroundTruthError(
            "no score column: expected 'manual_score', 'score' or '<rater>_score'")

    df[KEY_VIDEO] = df[KEY_VIDEO].str.strip()
    if level == "segment":
        seg = pd.to_numeric(df[KEY_SEGMENT], errors="coerce")
        if seg.isna().any() or (seg % 1 != 0).any():
            raise GroundTruthError("segment_index must be a whole number on every row")
        df[KEY_SEGMENT] = seg.astype(int)

    for r in raters:
        cells = df[r].fillna("").str.strip()
        values = pd.to_numeric(cells.replace("", None), errors="coerce")
        bad = df.loc[cells.ne("") & values.isna(), keys + [r]]
        if not bad.empty:
            raise GroundTruthError(
                f"non-numeric scores in '{r}': {bad.head(5).to_dict('records')}")
        df[r] = values

    dupes = df[df.duplicated(keys, keep=False)]
    if not dupes.empty:
        raise GroundTruthError(f"duplicate rows for {dupes[keys].head(5).to_dict('records')}")

    rated = df[raters].notna().any(axis=1)
    n_unrated = int((~rated).sum())
    df = df[rated].copy()
    df["manual"] = df[raters].mean(axis=1, skipna=True)
    df["n_raters"] = df[raters].notna().sum(axis=1)

    notes = []
    if n_unrated:
        notes.append(f"{n_unrated} row(s) had no score in any column and were skipped "
                     "(blank means not rated, never 0)")
    if len(raters) > 1 and (df["n_raters"] < len(raters)).any():
        notes.append("some rows were scored by fewer than all raters; their ground "
                     "truth is the mean of the raters who scored them")
    return GroundTruth(
        level=level, table=df[keys + raters + ["manual", "n_raters"]].reset_index(drop=True),
        raters=raters, sources=[str(f) for f in files], sha256=digest.hexdigest(),
        is_dummy=is_dummy, n_unrated_rows=n_unrated, notes=notes,
    )
