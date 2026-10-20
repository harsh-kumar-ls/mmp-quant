"""`mmpquant validate`: join manual and system scores, report agreement.

Order of operations, fixed in advance (16 Oct 2026):
  1. read the manual scores and the matching runs; refuse to mix runs scored
     under different configs;
  2. drop rows the system scored N/A (listed, never set to 0);
  3. split VIDEOS into tune / held-out (seeded, recorded);
  4. choose between the pre-registered variants on the tune videos only;
  5. headline = Spearman of the chosen variant on the held-out videos, with a
     95% bootstrap interval; nested leave-one-video-out beside it;
  6. exploratory: every variant on all videos (labelled as such -- choosing
     from this table would be tuning on the test data).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats
from .ground_truth import KEY_SEGMENT, KEY_VIDEO, GroundTruth
from .system_scores import load_system_scores, primary_variant

MIN_VIDEOS = 4
TARGET_RHO = 0.7


class ValidationError(RuntimeError):
    pass


@dataclass
class Settings:
    tune_fraction: float = 0.6
    seed: int = 0
    n_resamples: int = 10_000
    allow_mixed_config: bool = False


def _r(x: float | None, nd: int = 3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def run_validation(
    principle: str,
    gt: GroundTruth,
    runs_dir: Path,
    out_dir: Path,
    *,
    current_config: dict[str, Any] | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or Settings()
    keys = gt.keys
    video_ids = gt.table[KEY_VIDEO].unique().tolist()
    system = load_system_scores(runs_dir, principle, gt.level, video_ids, current_config)
    if system.config_mismatch and not s.allow_mixed_config:
        raise ValidationError(
            f"{len(system.config_mismatch)} run(s) were scored with a different config.yaml "
            f"than the current one: {system.config_mismatch[:8]}. Re-score them (scores from "
            "different settings are not comparable), or pass --allow-mixed-config to "
            "proceed and have it recorded.")

    variants = system.variants
    primary = primary_variant(principle)
    merged = gt.table.merge(system.table, on=keys, how="left", indicator=True)
    no_system = merged[merged["_merge"] == "left_only"][keys]
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    na_rows = merged[merged[primary].isna()][keys]
    data = merged[merged[primary].notna()].copy()
    for v in variants:
        data[v] = data[v].astype(float)

    n_videos = data[KEY_VIDEO].nunique()
    if n_videos < MIN_VIDEOS:
        raise ValidationError(
            f"only {n_videos} video(s) have both a manual and a system score; at least "
            f"{MIN_VIDEOS} are needed for any correlation to mean something")

    def summary(frame, column) -> dict[str, Any]:
        x, y = frame[column].to_numpy(float), frame["manual"].to_numpy(float)
        lo, hi, n_ok = stats.bootstrap_ci(x, y, frame[KEY_VIDEO].to_numpy(),
                                          n_resamples=s.n_resamples, seed=s.seed)
        return {"n_rows": len(frame), "n_videos": int(frame[KEY_VIDEO].nunique()),
                "spearman": _r(stats.spearman(x, y)), "spearman_p": _r(stats.spearman_p(x, y), 4),
                "spearman_ci95": [_r(lo), _r(hi)], "bootstrap_valid": n_ok,
                "pearson": _r(stats.pearson(x, y))}

    tune_ids, held_ids = stats.split_videos(data[KEY_VIDEO], s.tune_fraction, s.seed)
    tune = data[data[KEY_VIDEO].isin(tune_ids)]
    held = data[data[KEY_VIDEO].isin(held_ids)]
    chosen, tune_rhos = stats.choose_variant(tune.reset_index(drop=True), variants)
    oof, loo_choice = stats.nested_leave_one_video_out(data.reset_index(drop=True), variants)
    loo_frame = data.reset_index(drop=True).assign(_oof=oof)

    result: dict[str, Any] = {
        "principle": principle,
        "level": gt.level,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "is_dummy_ground_truth": gt.is_dummy,
        "ground_truth": {"sources": gt.sources, "sha256": gt.sha256, "raters": gt.raters,
                         "n_unrated_rows": gt.n_unrated_rows, "notes": gt.notes},
        "settings": {"tune_fraction": s.tune_fraction, "seed": s.seed,
                     "n_resamples": s.n_resamples, "allow_mixed_config": s.allow_mixed_config},
        "variants": variants,
        "primary_variant": primary,
        "excluded": {
            "no_system_score": no_system.to_dict("records"),
            "system_na": na_rows.to_dict("records"),
            "not_scored_reason": system.not_scored,
            "config_mismatch": system.config_mismatch,
        },
        "split": {"tune_videos": tune_ids, "held_out_videos": held_ids},
        "tune": {"spearman_by_variant": {k: _r(v) for k, v in tune_rhos.items()},
                 "chosen_variant": chosen},
        "held_out": {"variant": chosen, **summary(held, chosen),
                     "primary_for_reference": summary(held, primary)},
        "nested_loo": {**summary(loo_frame, "_oof"),
                       "variant_chosen_per_video": loo_choice,
                       "choice_counts": pd.Series(loo_choice).value_counts().to_dict()},
        "exploratory_all_videos": {v: summary(data, v) for v in variants},
        "target_spearman": TARGET_RHO,
        "share_of_rows_at_zero": {"system_primary": _r(float((data[primary] == 0).mean())),
                                  "manual": _r(float((data["manual"] == 0).mean()))},
    }
    if len(gt.raters) > 1:
        result["inter_rater"] = [{**d, "spearman": _r(d["spearman"])}
                                 for d in stats.inter_rater(data, gt.raters)]

    residuals = _residuals(data, chosen, keys, set(held_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(out_dir / "residuals.csv", index=False)
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2, default=str))
    _scatter(data, chosen, set(held_ids), residuals, result, out_dir / "scatter.png")
    (out_dir / "report.md").write_text(render_report(result, residuals))
    return result


def _residuals(data: pd.DataFrame, column: str, keys: list[str], held: set[str]) -> pd.DataFrame:
    """Per row: both scores, their ranks among all rows (average ranks for
    ties), and the rank difference -- the residual that matters for Spearman."""
    out = data[keys + ["manual", column]].rename(columns={column: "system"}).copy()
    out["manual_rank"] = out["manual"].rank(method="average")
    out["system_rank"] = out["system"].rank(method="average")
    out["rank_difference"] = out["system_rank"] - out["manual_rank"]
    out["split"] = np.where(out[KEY_VIDEO].isin(held), "held_out", "tune")
    out["variant"] = column
    return out.sort_values("rank_difference", key=abs, ascending=False).reset_index(drop=True)


def _scatter(data, column, held, residuals, result, path: Path) -> None:
    """System vs manual. One measure, so one hue (dataviz reference blue);
    tune vs held-out is told apart by marker shape and named in the legend.
    Only the five largest rank disagreements are labelled; points shared by
    several videos carry a count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = "#2a78d6", "#1f1f1e", "#6b6b67", "#e6e6e3"
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    is_held = data[KEY_VIDEO].isin(held)
    ax.scatter(data.loc[~is_held, "manual"], data.loc[~is_held, column], s=64, color=blue,
               edgecolor="white", linewidth=2, label="tune videos", zorder=3)
    ax.scatter(data.loc[is_held, "manual"], data.loc[is_held, column], s=64,
               facecolor="white", edgecolor=blue, linewidth=2, label="held-out videos", zorder=3)
    # Videos at exactly the same spot (common: many score 0) would hide each
    # other: show how many share a point, and merge their labels.
    stacked = data.groupby(["manual", column]).size()
    for (mx, sy), n in stacked[stacked > 1].items():
        ax.annotate(f"×{n}", (mx, sy), xytext=(-6, -11), textcoords="offset points",
                    fontsize=7.5, color=ink, ha="right")
    labels: dict[tuple[float, float], list[str]] = {}
    for _, row in residuals.head(5).iterrows():
        name = row[KEY_VIDEO] + (f" #{int(row[KEY_SEGMENT])}" if KEY_SEGMENT in row else "")
        labels.setdefault((row["manual"], row["system"]), []).append(name)
    for (mx, sy), names in labels.items():
        ax.annotate(", ".join(names), (mx, sy), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, color=muted)
    h = result["held_out"]
    ci = h["spearman_ci95"]
    ax.set_title(f"{result['principle'].capitalize()}: system vs manual  "
                 f"(held-out Spearman {h['spearman']}, 95% CI {ci[0]} to {ci[1]})",
                 fontsize=9.5, color=ink, loc="left")
    ax.set_xlabel("Manual score (mean of raters)", color=ink, fontsize=9)
    ax.set_ylabel(f"System score ({column})", color=ink, fontsize=9)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    if result["is_dummy_ground_truth"]:
        fig.text(0.5, 0.5, "TEST DATA — NOT A RESULT", ha="center", va="center",
                 fontsize=22, color="#d03b3b", alpha=0.35, rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt(x) -> str:
    return "—" if x is None else f"{x}"


def _fmt_p(p) -> str:
    return "—" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def render_report(r: dict[str, Any], residuals: pd.DataFrame) -> str:
    h, loo = r["held_out"], r["nested_loo"]
    lines = []
    if r["is_dummy_ground_truth"]:
        lines += [("> ⛔ **TEST DATA — NOT A RESULT.** The manual-score file is marked as "
                  "dummy/synthetic. Nothing below may be reported."), ""]
    lines += [
        f"# Validation — {r['principle']} ({r['level']} level)", "",
        (f"Created {r['created']}. Manual scores: {', '.join(r['ground_truth']['sources'][:3])}"
        f"{' …' if len(r['ground_truth']['sources']) > 3 else ''} "
        f"(sha256 `{r['ground_truth']['sha256'][:12]}`), raters: {', '.join(r['ground_truth']['raters'])}."),
        "",
        "## Headline: held-out videos", "",
        f"Variant **{h['variant']}**, chosen on the tune videos only.", "",
        "| | Spearman | 95% CI | p | Pearson | Videos | Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (f"| held-out | **{_fmt(h['spearman'])}** | {_fmt(h['spearman_ci95'][0])} to "
         f"{_fmt(h['spearman_ci95'][1])} | {_fmt_p(h['spearman_p'])} | {_fmt(h['pearson'])} | "
         f"{h['n_videos']} | {h['n_rows']} |"),
        (f"| nested leave-one-video-out | {_fmt(loo['spearman'])} | {_fmt(loo['spearman_ci95'][0])} to "
         f"{_fmt(loo['spearman_ci95'][1])} | {_fmt_p(loo['spearman_p'])} | {_fmt(loo['pearson'])} | "
         f"{loo['n_videos']} | {loo['n_rows']} |"),
        "",
        (f"Target: Spearman ≥ {r['target_spearman']}. The interval matters as much as the point "
        "estimate: with few videos it is wide, and a point estimate above the target with an "
        "interval reaching far below it is not evidence of agreement."),
        "",
        "Tune-split Spearman by variant: "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in r["tune"]["spearman_by_variant"].items())
        + ". Leave-one-out chose: "
        + ", ".join(f"{k} ×{v}" for k, v in loo["choice_counts"].items()) + ".",
        "",
    ]
    if "inter_rater" in r:
        lines += ["## Agreement between raters", "",
                  ("The honest ceiling for the system: it cannot be expected to agree with the "
                  "raters more than they agree with each other."), "",
                  "| Raters | Spearman | Rows |", "|---|---:|---:|"]
        lines += [f"| {' vs '.join(d['raters'])} | {_fmt(d['spearman'])} | {d['n']} |"
                  for d in r["inter_rater"]]
        lines.append("")
    lines += ["## Largest disagreements (by rank)", "",
              "| Video | Segment | Manual | System | Rank difference | Split |",
              "|---|---:|---:|---:|---:|---|"]
    for _, row in residuals.head(10).iterrows():
        seg = int(row[KEY_SEGMENT]) if KEY_SEGMENT in residuals.columns else "—"
        lines.append(f"| {row[KEY_VIDEO]} | {seg} | {row['manual']:.2f} | {row['system']:.2f} | "
                     f"{row['rank_difference']:+.1f} | {row['split']} |")
    ex = r["excluded"]
    lines += ["", "## Excluded", "",
              (f"- No system score: {len(ex['no_system_score'])} row(s) "
              f"{dict(list(ex['not_scored_reason'].items())[:6]) or ''}"),
              f"- System N/A (no narration): {len(ex['system_na'])} row(s) — excluded, never 0",
              f"- Runs scored with a different config: {ex['config_mismatch'] or 'none'}",
              f"- Rows with no manual score: {r['ground_truth']['n_unrated_rows']}", ""]
    lines += ["## Exploratory — every variant on all videos", "",
              ("**Not a result.** Choosing a variant from this table would be tuning on the "
              "held-out videos; it is here to show how much the choice matters."), "",
              "| Variant | Spearman | 95% CI | Pearson |", "|---|---:|---:|---:|"]
    for v, d in r["exploratory_all_videos"].items():
        lines.append(f"| {v}{' (primary)' if v == r['primary_variant'] else ''} | "
                     f"{_fmt(d['spearman'])} | {_fmt(d['spearman_ci95'][0])} to "
                     f"{_fmt(d['spearman_ci95'][1])} | {_fmt(d['pearson'])} |")
    z = r["share_of_rows_at_zero"]
    lines += ["", "## Method", "",
              (f"- Split by video, seed {r['settings']['seed']}, tune fraction "
              f"{r['settings']['tune_fraction']}: tune {len(r['split']['tune_videos'])}, "
              f"held-out {len(r['split']['held_out_videos'])} videos."),
              (f"- 95% percentile bootstrap, {r['settings']['n_resamples']} resamples, resampling "
              "videos (segments of one video are not independent)."),
              (f"- Ties: {z['system_primary']:.0%} of rows score 0 on the system, "
              f"{z['manual']:.0%} on the manual scores; Spearman uses average ranks."),
              "- Variants were fixed before any manual score was seen (CLAUDE.md §2, §2b).",
              "- Files: validation.json, residuals.csv, scatter.png.", ""]
    for note in r["ground_truth"]["notes"]:
        lines.append(f"> {note}")
    return "\n".join(lines).rstrip() + "\n"
