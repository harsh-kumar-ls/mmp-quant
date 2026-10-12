"""Human-readable Markdown report for one scored video.

`result.json` is the machine record and is written regardless; this module is
the version a person reads. Build order step 3.

The report has one job beyond restating the score: make the number falsifiable.
A reader who disagrees with 33.5 should be able to find the segment that drove
it, read the on-screen text and the narration side by side, and say "that is not
redundancy" — without opening a JSON file or trusting the pipeline. So the
evidence is quoted in full rather than summarised, exclusions are counted rather
than hidden, and every threshold that shaped the number is printed at the end.

Nothing here recomputes anything. The report is a view over `result.json`; if a
number appears here it appeared there first.
"""

from __future__ import annotations

from typing import Any

BAR_WIDTH = 24

# Windows are dropped for several distinct reasons and the difference matters
# when reading a low score: a video excluded for "only labels on screen" was
# assessed and found non-redundant, whereas one excluded for "no narration"
# was not assessed at all.
EXCLUSION_LABELS = {
    "no_onscreen_text": "no on-screen text of any kind",
    "signaling_exception_short_onscreen_text": (
        "on-screen sentences too short to count (signaling exception)"
    ),
    "no_narration": "no narration to compare against",
}


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.2f}"


def _timecode(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _bar(score: float | None, width: int = BAR_WIDTH) -> str:
    """A fixed-scale bar, 0-100. Fixed so segments compare across videos."""
    if score is None:
        return "—"
    filled = round(max(0.0, min(100.0, score)) / 100.0 * width)
    return "█" * filled + "·" * (width - filled)


def _cell(text: str, limit: int = 2000) -> str:
    """Make free text safe inside a Markdown table cell.

    Newlines and pipes both break table rendering. OCR text is full of both, so
    it is flattened rather than dropped — the reader needs the words, not the
    original line breaks, and the untouched text is in `result.json`.
    """
    flat = " ".join(text.split()).replace("|", "\\|")
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat or "_(none)_"


def _reliability_banner(rel: dict[str, Any]) -> list[str]:
    if not rel or rel.get("reliability") == "ok":
        return []
    if rel.get("reliability") == "no_text_detected":
        return [
            "> **No on-screen text was detected anywhere in this video.**",
            f"> {rel.get('note', '')}",
            "",
        ]
    share = rel.get("detection_share")
    recov = rel.get("detection_recovery_rate")
    measured = (
        f"> Text detected across **{share:.0%}** of the video "
        f"(by {rel.get('detection_share_basis', 'unknown basis')}); "
        f"**{recov:.0%}** of detected regions could be read."
    )
    return [
        "> ## ⚠ OCR UNRELIABLE — this score is not trustworthy",
        ">",
        f"> {rel.get('note', '')}",
        ">",
        measured,
        ">",
        (
            "> A low score here means the text could not be read, **not** that the "
            "video is free of redundancy. Inspect frames before reporting this number."
        ),
        "",
    ]


def _header(payload: dict[str, Any]) -> list[str]:
    res = payload["result"]
    vid = payload["video_id"]
    dur = payload["duration"]
    score = res["score"]

    lines = [
        f"# Redundancy report — `{vid}`",
        "",
        (
            f"**Score: {_fmt_score(score)} / 100**  "
            f"({'higher = more redundant = worse per Mayer' if score is not None else res.get('status')})"
        ),
        "",
    ]
    lines += _reliability_banner(payload.get("ocr_recognition_reliability", {}))

    if score is None:
        lines += [
            "This video could not be scored.",
            "",
            f"> {res.get('na_reason') or 'no reason recorded'}",
            "",
            (
                "**N/A is not zero.** Zero would mean the video was assessed and found "
                "free of redundancy; N/A means it could not be assessed at all."
            ),
            "",
        ]

    lines += [
        "| | |",
        "|---|---|",
        f"| Source | `{payload['video']}` |",
        f"| Duration | {_timecode(dur)} ({dur:.1f} s) |",
        f"| Segments | {len(payload['segments'])} × {payload['segment_length_s']:.0f} s |",
        f"| Windows scored | {res['n_windows_qualifying']} of {res['n_windows_total']} |",
    ]
    if payload.get("coverage") is not None and score is not None:
        lines.append(f"| Time with an on-screen sentence to assess | {payload['coverage']:.0%} |")
    lines += [""]
    for w in res.get("warnings", []):
        lines += [f"> ⚠ {w}", ""]
    return lines


def _segment_timeline(payload: dict[str, Any]) -> list[str]:
    lines = [
        "## Segment timeline",
        "",
        (
            "Segments are the unit the human raters score, so this is the table to "
            "compare against their sheets."
        ),
        "",
        "| Segment | Time | Score | | Verbatim | Semantic | Windows scored |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for s in payload["segments"]:
        v = s.get("verbatim")
        m = s.get("semantic")
        lines.append(
            f"| {s['index']} | {s['timecode']} | {_fmt_score(s['score'])} | "
            f"`{_bar(s['score'])}` | "
            f"{'—' if v is None else f'{v:.3f}'} | "
            f"{'—' if m is None else f'{m:.3f}'} | "
            f"{s['n_windows_qualifying']}/{s['n_windows']} |"
        )
    lines.append("")
    return lines


def _evidence_block(w: dict[str, Any]) -> list[str]:
    """One qualifying window, with the two texts that produced its score."""
    c = w.get("components", {})
    onscreen = w.get("ocr_sentence_text", "")
    n_labels = int(c.get("n_label_elements", 0))

    head = (
        f"**{_timecode(w['start'])}–{_timecode(w['end'])}** — "
        f"score {_fmt_score(w['score'])} "
        f"(verbatim {c.get('verbatim', 0):.3f}, semantic {c.get('semantic', 0):.3f})"
    )
    ev = w.get("evidence") or {}
    restated = [r["sentence"] for r in ev.get("restated_sentences") or []]
    shared = ev.get("shared_content_words") or []
    if restated:
        head += " — restates: " + "; ".join(f"“{s}”" for s in restated)
    if shared:
        head += f" (narration repeated: {', '.join(shared)})"
    lines = [
        head,
        "",
        "| On-screen text (scored) | Narration |",
        "|---|---|",
        f"| {_cell(onscreen)} | {_cell(w.get('asr_text', ''))} |",
        "",
    ]
    if n_labels:
        lines += [
            (
                f"_{n_labels} further on-screen element(s) were classified as labels "
                "or key terms and excluded from the comparison — signaling, not "
                "redundancy._"
            ),
            "",
        ]
    return lines


def _worst_segments(payload: dict[str, Any], top_n: int) -> list[str]:
    scored = [s for s in payload["segments"] if s["score"] is not None]
    # A floor-scoring video has segments that were assessed and found empty. Its
    # "worst offenders" are three subsections saying nothing was found, which
    # buries the one fact that matters.
    if scored and max(s["score"] for s in scored) == 0.0:
        scored = []
    if not scored:
        return [
            "## Worst-offending segments",
            "",
            "No segment carried measurable redundancy.",
            "",
        ]

    worst = sorted(scored, key=lambda s: s["score"], reverse=True)[:top_n]
    by_index = {w["index"]: w for w in payload["result"]["windows"]}

    lines = [
        f"## Worst-offending segments (top {len(worst)})",
        "",
        (
            "The evidence behind the highest scores. If the score is wrong, it is "
            "wrong here — read the two columns and judge whether the narration is "
            "duplicating the screen."
        ),
        "",
    ]
    for s in worst:
        lines += [
            f"### Segment {s['index']} — {s['timecode']} — score {_fmt_score(s['score'])}",
            "",
        ]
        qualifying = [
            by_index[i]
            for i in s.get("window_indices", [])
            if i in by_index and by_index[i].get("qualifies")
        ]
        if not qualifying:
            lines += ["_No qualifying window in this segment._", ""]
            continue
        for w in sorted(qualifying, key=lambda w: w["score"] or 0, reverse=True):
            lines += _evidence_block(w)
    return lines


def _exclusions(payload: dict[str, Any]) -> list[str]:
    windows = payload["result"]["windows"]
    excluded = [w for w in windows if not w.get("qualifies")]
    lines = [
        "## Windows not scored",
        "",
    ]
    if not excluded:
        lines += ["Every window was scored.", ""]
        return lines

    counts: dict[str, int] = {}
    for w in excluded:
        counts[w.get("exclusion_reason") or "unrecorded"] = (
            counts.get(w.get("exclusion_reason") or "unrecorded", 0) + 1
        )

    lines += [
        (
            f"{len(excluded)} of {len(windows)} windows carried nothing measurable. "
            "This is not a failure — it is how the operative definition excludes "
            "signaling — but it determines how much of the video the score rests on."
        ),
        "",
        "| Reason | Windows |",
        "|---|---:|",
    ]
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {EXCLUSION_LABELS.get(reason, reason)} | {n} |")
    lines.append("")
    return lines


def _basis_phrase(rel: dict[str, Any]) -> str:
    return {
        "duration": "weighted by duration",
        "group_count": "counted over deduplicated frame groups",
    }.get(rel.get("detection_share_basis"), "basis not recorded")


def _basis_caveat(rel: dict[str, Any]) -> list[str]:
    """Say plainly when the detection share is not a share of the video.

    The figure is a statement about how much of the video showed text only when
    it is weighted by duration. Counted over deduplicated groups it is not, and
    the report must not imply otherwise: a static board held for a minute is one
    group against a hundred frames of camera movement, so the group-counted
    figure understates exactly the static, text-bearing content that matters.
    """
    if rel.get("detection_share_basis") == "duration":
        return [
            (
                "The share above is weighted by duration rather than by deduplicated "
                "frame group, so it is a genuine share of the running time."
            ),
            "",
        ]
    return [
        (
            "> ⚠ **This share is counted over deduplicated frame groups, not weighted "
            "by duration, so it is not a share of the running time** — a static board "
            "held for a minute counts once against a hundred frames of camera "
            "movement. This run predates the duration-weighting fix; re-run it with "
            "`--no-resume` for a figure that can be read as a share of the video."
        ),
        "",
    ]


def _aggregation_note(payload: dict[str, Any]) -> list[str]:
    """The assessed-only average and coverage, beside the reported score.

    score = assessed_score × coverage, so the two differ exactly when part of
    the video had no on-screen sentence to assess. Shown only when they differ,
    since printing an identical second number on every fully covered video
    would just be noise. Runs from before 08 Oct 2026 carry neither figure.
    """
    assessed = payload.get("assessed_score")
    score = payload["result"]["score"]
    coverage = payload.get("coverage")
    if assessed is None or score is None or coverage is None or abs(assessed - score) < 0.1:
        return []
    return [
        (
            f"> An on-screen sentence could be assessed in **{coverage:.0%}** of this video. "
            f"Where it could, redundancy averaged **{assessed:.2f}**; the rest of the time "
            f"there was nothing on screen to duplicate the narration, which counts as 0, "
            f"giving **{score:.2f}** for the whole video "
            f"({assessed:.2f} × {coverage:.4f} = {assessed * coverage:.2f}). "
            "See `segment_scores.csv`."
        ),
        "",
    ]


def _methodology(payload: dict[str, Any]) -> list[str]:
    res = payload["result"]
    p = res["params"]
    cfg = payload.get("config", {})
    rel = payload.get("ocr_recognition_reliability", {})
    w = p["weights"]
    norm = p.get("text_normalisation", {})

    lines = [
        "## Methodology appendix",
        "",
        "Every threshold that shaped the number above, for this run specifically.",
        "",
        "### Score",
        "",
        "```",
        "gate          : the narration must say >= 2 of the on-screen content words,",
        "                and more of them than narration elsewhere in the video does",
        "                (otherwise the window scores 0 -- text on screen is not",
        "                redundancy by being there)",
        "",
        f"window score  = 100 × ({w['verbatim']} × ROUGE-L F1 over content words, above chance",
        f"{' ' * 24}+ {w['semantic']} × cosine similarity, above chance)",
        "  above chance(x) = max(0, (x - c) / (1 - c)), c = median of the same measure",
        "                    against narration > 30 s away in the same video",
        "",
        "segment score = sum(window score × duration) ÷ duration of all windows in the",
        "                segment; a window with nothing to assess counts as 0",
        "",
        "video score   = the same over the whole video",
        "              = duration-weighted mean of the segment scores",
        "```",
        "",
        (
            "Only text classified as a **sentence** is compared. Labels, key terms and "
            "headlines are signaling under Mayer, not redundancy, and are removed from "
            "the compared text entirely so they cannot raise the score even indirectly. "
            "Classification turns on grammatical completeness — the presence of a "
            "finite verb — not on length."
        ),
        "",
        *_aggregation_note(payload),
        "### Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Window size | {cfg.get('windowing', {}).get('size_s', '?')} s |",
        f"| Segment length | {payload['segment_length_s']} s |",
        f"| Alignment tolerance | ±{cfg.get('alignment', {}).get('tolerance_s', '?')} s |",
        f"| Frame sampling | {cfg.get('media', {}).get('frame_fps', '?')} fps |",
        f"| Min on-screen sentence words | {p['min_onscreen_words']} |",
        f"| Min narration words | {p['min_narration_words']} |",
        f"| Verbatim weight | {w['verbatim']} |",
        f"| Semantic weight | {w['semantic']} |",
        f"| Embedding model | `{p['embedding_model']}` |",
        (
            f"| ASR | `{cfg.get('asr', {}).get('backend', '?')}` "
            f"/ `{cfg.get('asr', {}).get('model', '?')}` |"
        ),
        f"| OCR | `{cfg.get('ocr', {}).get('backend', '?')}` |",
        f"| Text normalisation | {', '.join(k for k, v in norm.items() if v) or 'none'} |",
        f"| Config | `{payload.get('config_path', '?')}` |",
        "",
        "### OCR reliability",
        "",
        "| | |",
        "|---|---|",
        f"| Verdict | `{rel.get('reliability', '?')}` |",
        (
            f"| Text detected across | {rel.get('detection_share', 0):.1%} of the video "
            f"({_basis_phrase(rel)}) |"
        ),
        (
            f"| Detected regions read | {rel.get('n_regions_recovered', 0)} of "
            f"{rel.get('n_regions_detected', 0)} |"
        ),
        "",
        (
            "Detection and recognition fail independently. Text that is detected but "
            "unreadable — handwriting, a low-contrast board — must never be reported "
            "as text-free, because that turns the most redundant videos into apparently "
            "ideal ones."
        ),
        "",
        *_basis_caveat(rel),
        "### Scale",
        "",
        (
            "**0** means assessed and found free of redundancy. **N/A** means not "
            "assessable — no narration anywhere to compare against. The two are never "
            "interchangeable."
        ),
        "",
    ]
    return lines


def _time_ranges(starts: list[float], hop: float = 1.0) -> list[tuple[float, float]]:
    """Merge frame start times into contiguous [start, end) ranges."""
    ranges: list[list[float]] = []
    for t in sorted(starts):
        if ranges and t <= ranges[-1][1] + 1e-6:
            ranges[-1][1] = t + hop
        else:
            ranges.append([t, t + hop])
    return [(a, b) for a, b in ranges]


def _fmt_ranges(ranges: list[tuple[float, float]]) -> str:
    return ", ".join(f"{_timecode(a)}–{_timecode(b)}" for a, b in ranges)


def _coherence(payload: dict[str, Any]) -> list[str]:
    """The coherence section: score, parts, evidence, method. Empty if the run
    did not score coherence (earlier runs, or disabled in config)."""
    coh = payload.get("coherence")
    if not coh:
        return []
    res = coh["result"]
    p = res["params"]
    w = p["weights"]
    lines = ["## Coherence", ""]
    if res["score"] is None:
        return lines + [
            f"**Score: N/A** — {res.get('na_reason') or 'no reason recorded'}", "",
            "**N/A is not zero.**", "",
        ]

    parts = coh["parts"]
    lines += [
        (f"**Score: {res['score']:.2f} / 100**  (higher = more extraneous material = "
         "worse per Mayer)"),
        "",
        "| Part | Score | Weight |",
        "|---|---:|---:|",
        (f"| Background music under narration | {parts['background_music']:.2f} | "
         f"{w['background_music']} |"),
        (f"| Off-topic narration{' (experimental — not in the score)' if not w['off_topic'] else ''} | "
         f"{parts['off_topic']:.2f} | {w['off_topic']} |"),
        "",
        (f"{w['background_music']} × {parts['background_music']:.2f} + {w['off_topic']} × "
         f"{parts['off_topic']:.2f} = **{res['score']:.2f}**"),
        "",
    ]
    alt = coh.get("alternative")
    if alt and alt.get("score") is not None:
        aw = alt["weights"]
        lines += [
            (f"Pre-registered alternative ({aw['background_music']} / {aw['off_topic']}, off-topic "
             f"included): **{alt['score']:.2f}** — kept for comparison against the professor's "
             "scores; not the reported score."),
            "",
        ]
    if not w["off_topic"]:
        lines += [
            ("> The off-topic part is **experimental** and carries weight 0: read by hand on "
             "this corpus, most windows it flags are short remarks about the board or on-topic "
             "talk in everyday words, not digressions. It is shown so it can be checked."),
            "",
        ]
    for warning in res.get("warnings", []):
        lines += [f"> ⚠ {warning}", ""]

    lines += [
        "### Coherence segment timeline", "",
        "| Segment | Time | Score | | Music share | Off-topic share | Windows with narration |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for seg in coh["segments"]:
        sp = seg.get("parts") or {}
        bm, ot = sp.get("background_music"), sp.get("off_topic")
        lines.append(
            f"| {seg['index']} | {seg['timecode']} | {_fmt_score(seg['score'])} | "
            f"`{_bar(seg['score'])}` | {'—' if bm is None else f'{bm:.2f}'} | "
            f"{'—' if ot is None else f'{ot:.2f}'} | "
            f"{seg['n_windows_qualifying']}/{seg['n_windows']} |"
        )
    lines.append("")

    windows = res["windows"]
    under = [t for x in windows for t in x["evidence"]["music"]["music_under_speech_at"]]
    alone = [t for x in windows for t in x["evidence"]["music"]["music_without_speech_at"]]
    heard: dict[str, int] = {}
    for x in windows:
        for label in x["evidence"]["music"]["heard"]:
            heard[label] = heard.get(label, 0) + 1
    lines += ["### Background music", ""]
    if under:
        lines += [
            f"Music under narration at: {_fmt_ranges(_time_ranges(under))}.",
            "",
            f"The sound classifier heard: {', '.join(list(heard)[:6])}.",
            "",
        ]
    else:
        lines += ["No music detected under narration.", ""]
    if alone:
        lines += [
            (f"Music with nobody speaking at {_fmt_ranges(_time_ranges(alone))} — "
             "recorded, not counted (it does not compete with narration)."),
            "",
        ]

    off = [x for x in windows if (x["evidence"]["off_topic"] or {}).get("off_topic")]
    lines += ["### Off-topic narration", ""]
    ev0 = next((x["evidence"]["off_topic"] for x in windows
                if x["evidence"]["off_topic"].get("typical") is not None), None)
    if ev0 and ev0.get("cut") is not None:
        lines += [
            (f"Typical relevance of this video's narration to its own topic: "
             f"{ev0['typical']:.3f}; unrelated everyday text: {ev0['unrelated']:.3f}; "
             f"a window is off-topic below {ev0['cut']:.3f}."),
            "",
        ]
    if off:
        lines += ["| Time | Relevance | Narration |", "|---|---:|---|"]
        for x in off:
            ev = x["evidence"]["off_topic"]
            lines.append(
                f"| {_timecode(x['start'])}–{_timecode(x['end'])} | {ev['relevance']:.3f} | "
                f"{_cell(x['evidence']['narration'], 600)} |"
            )
        lines.append("")
    else:
        lines += ["No window's narration was judged off-topic.", ""]

    lines += [
        "### Coherence method", "",
        "```",
        ("background music = share of the window's spoken seconds with "
         f"{'/'.join(p['music_classes'])} probability >= {p['music_threshold']}"),
        f"                   ({p['audio_model']}); music with nobody speaking is not counted",
        ("off-topic        = 1 if the window's narration is nearer (cosine, "
         f"{p['embedding_model'].split('/')[-1]}) to"),
        "                   unrelated everyday text than to the video's typical narration:",
        f"                   relevance < unrelated + {p['decision_point']} × (typical − unrelated)",
        f"                   (windows with < {p['min_narration_words']} spoken words are not judged)",
        (f"window score     = 100 × ({w['background_music']} × background music + "
         f"{w['off_topic']} × off-topic)"),
        "segment / video  = duration-weighted mean over all windows; a window with no",
        "                   narration counts as 0 (same rule as redundancy)",
        "```",
        "",
        (f"Unrelated reference: `{p['reference_texts']}` ({p['n_reference_texts']} everyday "
         "paragraphs, fixed and versioned). Limitations: the off-topic test is relative to "
         "the video itself, so a video off-topic throughout cannot be detected, and a "
         "legitimate change of subtopic can look like a digression."),
        "",
    ]
    return lines


def render_markdown(payload: dict[str, Any], *, top_n: int = 3) -> str:
    """Render `result.json` as a Markdown report."""
    parts: list[str] = []
    parts += _header(payload)
    parts += _segment_timeline(payload)
    parts += _worst_segments(payload, top_n)
    parts += _exclusions(payload)
    parts += _coherence(payload)
    parts += _methodology(payload)
    return "\n".join(parts).rstrip() + "\n"
