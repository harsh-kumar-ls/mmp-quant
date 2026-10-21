"""The regression loop (20 Oct 2026): every change is checked against the
user's own judgements and against the last accepted scores, before it is
reported as done.

Why it exists: the code and its unit tests share an author, so tests check
that the code does what was intended -- not that what was intended is right.
The problems the user kept finding by hand (Microlearning at 41, the first
caption-stitching gluing chalkboards, ...) were of that second kind.

Two independent checks, both run on real videos:

  1. Golden expectations (tests/golden/expectations.yaml): facts the user has
     stated about specific videos. A failure means the system now contradicts
     the user.
  2. Score changes against the accepted baseline (tests/golden/baseline.json):
     every video whose score moved is listed, window by window, with the
     evidence -- so side effects cannot pass silently. A change is accepted
     only with `--accept`, after the user has looked at it.

Neither uses the professor's manual scores: those are for validation only.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GOLDEN_DIR = Path("tests/golden")
EXPECTATIONS = GOLDEN_DIR / "expectations.yaml"
BASELINE = GOLDEN_DIR / "baseline.json"
CACHED_ARTIFACTS = ("transcript.json", "ocr.json", "audio_events.json", "audio.wav")
TOLERANCE = 0.01  # score points: below this a "change" is rounding


@dataclass
class CheckResult:
    id: str
    status: str  # draft | confirmed
    passed: bool
    detail: str
    source: str = ""


# ------------------------------------------------------------------ expectations


def _get(payload: dict, principle: str, field: str = "score"):
    if principle == "redundancy":
        return payload["result"]["score"] if field == "score" else payload.get(field)
    coh = payload.get("coherence") or {}
    if field == "score":
        return (coh.get("result") or {}).get("score")
    if field.startswith("part:"):
        return (coh.get("parts") or {}).get(field[5:])
    return coh.get(field)


def _windows(payload: dict, principle: str) -> list[dict]:
    if principle == "redundancy":
        return payload["result"]["windows"]
    return payload["coherence"]["result"]["windows"]


def _music_seconds(payload: dict) -> list[float]:
    return [t for w in _windows(payload, "coherence")
            for t in w["evidence"]["music"]["music_under_speech_at"]]


_OPS: dict[str, Callable[[float, float], bool]] = {
    "==": lambda a, b: abs(a - b) <= 1e-9, ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
}


def check_one(c: dict, results: dict[str, dict]) -> tuple[bool, str]:
    t = c["type"]
    need = [c["video"]] if "video" in c else list(c.get("videos", []))
    need += list(c.get("than", []))
    missing = [v for v in need if v not in results]
    if missing:
        return False, f"no run for {missing}"

    if t == "score":
        v = _get(results[c["video"]], c["principle"], c.get("field", "score"))
        if v is None:
            return False, f"{c['video']} has no {c.get('field', 'score')} (N/A)"
        return _OPS[c["op"]](v, c["value"]), f"{c['video']} {c.get('field', 'score')} = {v} (need {c['op']} {c['value']})"

    if t == "highest":
        mine = _get(results[c["video"]], c["principle"])
        others = {k: _get(p, c["principle"]) for k, p in results.items() if k != c["video"]}
        top = max(others.items(), key=lambda kv: kv[1] or -1)
        return (mine or -1) > (top[1] or -1), f"{c['video']} = {mine}; next highest {top[0]} = {top[1]}"

    if t == "window_mean":
        ws = [w for w in _windows(results[c["video"]], c["principle"])
              if c["from_s"] <= w["start"] < c["to_s"]]
        if not ws:
            return False, "no windows in range"
        mean = sum((w["score"] or 0.0) for w in ws) / len(ws)
        return _OPS[c["op"]](mean, c["value"]), (
            f"mean of {len(ws)} windows {c['from_s']}-{c['to_s']} s = {mean:.2f} "
            f"(need {c['op']} {c['value']}); lowest {min((w['score'] or 0) for w in ws):.2f}")

    if t == "window_each":
        ws = [w for w in _windows(results[c["video"]], c["principle"])
              if c["from_s"] <= w["start"] < c["to_s"]]
        if not ws:
            return False, "no windows in range"
        bad = [(int(w["start"]), round(w["score"] or 0.0, 2)) for w in ws
               if not _OPS[c["op"]](w["score"] or 0.0, c["value"])]
        return not bad, (f"all {len(ws)} windows {c['from_s']}-{c['to_s']} s {c['op']} {c['value']}"
                         if not bad else f"{len(bad)} of {len(ws)} windows fail: {bad}")

    if t == "never_sentence":
        bad = sorted({e["text"].replace("\n", " ")[:60]
                      for w in _windows(results[c["video"]], "redundancy")
                      for e in w["ocr_elements"]
                      if c["text_contains"].lower() in e["text"].lower() and e["kind"] == "sentence"})
        return not bad, (f"no element containing '{c['text_contains']}' classed as a sentence"
                         if not bad else f"classed as sentence: {bad[:3]}")

    if t == "text_never_counted":
        k = "".join(ch for ch in c["text"].lower() if ch.isalnum())
        hits = sorted({int(w["start"]) for w in _windows(results[c["video"]], "redundancy")
                       for r in (w.get("evidence") or {}).get("restated_sentences") or []
                       if k in "".join(ch for ch in r["sentence"].lower() if ch.isalnum())})
        return not hits, ("never counted" if not hits else f"counted as redundancy at {hits} s")

    if t == "question_not_restated":
        hits = [w["start"] for w in _windows(results[c["video"]], "redundancy")
                for chk in (w.get("evidence") or {}).get("sentence_checks") or []
                if chk["sentence"].strip().lower() == c["question"].lower() and chk["passed"]]
        return not hits, ("never counted" if not hits else f"counted as restated at {hits} s")

    if t == "music_present":
        none = [v for v in c["videos"] if not _music_seconds(results[v])]
        return not none, ("music under speech found in all" if not none
                          else f"no music under speech found in {none}")

    if t == "music_within":
        secs = _music_seconds(results[c["video"]])
        lo, hi = c["from_s"] - c["tolerance_s"], c["to_s"] + c["tolerance_s"]
        outside = [s for s in secs if not lo <= s < hi]
        ok = bool(secs) and not outside
        return ok, (f"{len(secs)} s of music under speech, "
                    f"{'all' if not outside else f'{len(outside)} s NOT'} within "
                    f"{c['from_s']}-{c['to_s']} s ±{c['tolerance_s']}"
                    + (f" (outside at {outside[:6]})" if outside else ""))

    if t == "no_music_elsewhere":
        skip = set(c["music_videos"]) | set(c.get("not_yet_listened", []))
        over = {k: _get(p, "coherence", "part:background_music") for k, p in results.items()
                if k not in skip and p.get("coherence")}
        over = {k: v for k, v in over.items() if (v or 0) > c["max_part"]}
        pending = (f"; not yet listened to: {c['not_yet_listened']}"
                   if c.get("not_yet_listened") else "")
        return not over, ("no music elsewhere" if not over else f"music part above "
                          f"{c['max_part']} in {over}") + pending

    if t == "louder":
        def loud(v):
            return ((results[v].get("coherence") or {}).get("music_loudness") or {}).get("median_db")
        mine = loud(c["video"])
        others = {v: loud(v) for v in c["than"]}
        ok = mine is not None and all(o is not None and mine > o for o in others.values())
        return ok, f"{c['video']} {mine} dB vs {others}"

    return False, f"unknown check type {t!r}"


def _videos_of(c: dict) -> set[str]:
    return ({c["video"]} if "video" in c else set()) | set(c.get("videos", [])) | set(
        c.get("than", []))


def check_expectations(results: dict[str, dict], path: Path = EXPECTATIONS,
                       only: set[str] | None = None) -> list[CheckResult]:
    """`only`: when checking a subset of videos, skip checks about others
    (corpus-wide checks such as 'highest' still need the full set)."""
    spec = yaml.safe_load(path.read_text())
    out = []
    for c in spec.get("checks", []):
        if only is not None:
            vids = _videos_of(c)
            if not vids or not vids <= only:
                continue
        try:
            ok, detail = check_one(c, results)
        except Exception as exc:  # noqa: BLE001 -- a malformed check must fail loudly, not pass
            ok, detail = False, f"check error: {exc!r}"
        out.append(CheckResult(c["id"], c.get("status", "draft"), ok, detail, c.get("source", "")))
    return out


# ------------------------------------------------------------------ snapshot & diff


def snapshot(results: dict[str, dict]) -> dict[str, Any]:
    snap = {}
    for vid, p in sorted(results.items()):
        coh = p.get("coherence") or {}
        snap[vid] = {
            "redundancy": p["result"]["score"],
            "redundancy_assessed": p.get("assessed_score"),
            "redundancy_windows": [w["score"] for w in p["result"]["windows"]],
            "coherence": (coh.get("result") or {}).get("score"),
            "coherence_parts": coh.get("parts"),
            "coherence_windows": [w["score"] for w in (coh.get("result") or {}).get("windows", [])],
        }
    return snap


def _moved(a, b) -> bool:
    if a is None or b is None:
        return a != b
    return abs(a - b) > TOLERANCE


def diff(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for vid in sorted(set(baseline) | set(current)):
        b, c = baseline.get(vid), current.get(vid)
        if b is None or c is None:
            changes.append({"video": vid, "kind": "added" if b is None else "removed"})
            continue
        for key in ("redundancy", "redundancy_assessed", "coherence"):
            if _moved(b.get(key), c.get(key)):
                wkey = "coherence_windows" if key == "coherence" else "redundancy_windows"
                bw, cw = b.get(wkey, []), c.get(wkey, [])
                windows = [{"index": i, "before": x, "after": y}
                           for i, (x, y) in enumerate(zip(bw, cw, strict=False)) if _moved(x, y)]
                changes.append({"video": vid, "kind": key, "before": b.get(key),
                                "after": c.get(key), "windows": windows})
    return changes


def load_results(runs_dir: Path, videos: list[str]) -> dict[str, dict]:
    out = {}
    for v in videos:
        f = runs_dir / v / "result.json"
        if f.exists():
            out[v] = json.loads(f.read_text())
    return out


# ------------------------------------------------------------------ re-scoring


def rescore_all(videos: dict[str, Path], runs_dir: Path, scratch: Path, cfg,
                log: Callable[[str], None] = print) -> dict[str, str]:
    """Re-score every video in ONE process (models loaded once) into `scratch`,
    reusing each run's cached transcript / OCR / audio events. Never runs ASR
    or OCR: a video without cached artifacts is reported, not scored."""
    from .overlap import SentenceTransformerEmbedder
    from .pipeline import run_pipeline

    embedder = SentenceTransformerEmbedder(str(cfg.get("scoring.redundancy.embedding_model")))
    problems = {}
    for i, (vid, path) in enumerate(sorted(videos.items()), 1):
        src = runs_dir / vid
        missing = [a for a in ("transcript.json", "ocr.json") if not (src / a).exists()]
        if missing:
            problems[vid] = f"not scored yet ({', '.join(missing)} missing) — run `mmpquant score` first"
            continue
        dst = scratch / vid
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for a in CACHED_ARTIFACTS:
            if (src / a).exists():
                (dst / a).symlink_to((src / a).resolve())
        try:
            run_pipeline(path, dst, cfg, embedder=embedder)
            log(f"[{i}/{len(videos)}] {vid}")
        except Exception as exc:  # noqa: BLE001 -- report every failure, keep going
            problems[vid] = f"scoring failed: {exc!r}"
            log(f"[{i}/{len(videos)}] {vid} FAILED: {exc!r}")
    return problems


def accept(scratch: Path, runs_dir: Path, videos: list[str]) -> None:
    """Make the re-scored results the real ones (runs/<video>/)."""
    for vid in videos:
        for f in (scratch / vid).iterdir():
            if f.is_symlink():
                continue
            target = runs_dir / vid / f.name
            if f.is_file():
                shutil.copy2(f, target)


# ------------------------------------------------------------------ report


def render(checks: list[CheckResult], changes: list[dict], problems: dict[str, str],
           current: dict[str, Any], baseline_exists: bool) -> str:
    failed = [c for c in checks if not c.passed]
    lines = ["# Regression check", "",
             f"**Expectations:** {len(checks) - len(failed)}/{len(checks)} pass"
             + ("" if not failed else f" — **{len(failed)} FAIL**"),
             "**Score changes vs accepted baseline:** "
             + ("no baseline yet" if not baseline_exists else f"{len(changes)}"),
             f"**Videos not scored:** {len(problems)}", ""]
    lines += ["## Expectations (the user's judgements)", "",
              "| | Id | Status | Result | Source |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {'✅' if c.passed else '❌'} | {c.id} | {c.status} | "
                     f"{c.detail.replace('|', '/')} | {c.source.replace('|', '/')[:120]} |")
    if changes:
        lines += ["", "## Score changes (need review before --accept)", "",
                  "| Video | What | Before | After | Windows that moved |", "|---|---|---:|---:|---|"]
        for ch in changes:
            if ch["kind"] in ("added", "removed"):
                lines.append(f"| {ch['video']} | {ch['kind']} | | | |")
                continue
            moved = ", ".join(f"#{w['index']} {w['before']}→{w['after']}" for w in ch["windows"][:8])
            more = f" (+{len(ch['windows']) - 8})" if len(ch["windows"]) > 8 else ""
            lines.append(f"| {ch['video']} | {ch['kind']} | {ch['before']} | {ch['after']} | "
                         f"{moved}{more} |")
    if problems:
        lines += ["", "## Not scored", ""] + [f"- {k}: {v}" for k, v in problems.items()]
    lines += ["", "## Current scores", "", "| Video | Redundancy | Coherence |", "|---|---:|---:|"]
    for vid, s in current.items():
        lines.append(f"| {vid} | {s['redundancy']} | {s['coherence']} |")
    return "\n".join(lines) + "\n"
