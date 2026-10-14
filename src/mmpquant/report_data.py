"""`result.json` as structured data for a UI, instead of as Markdown.

`report.py` turns a run's payload into text. This module turns the same
payload into a plain dict of primitives (no formatting, no line-wrapping), so a
frontend can lay the evidence out as cards and charts rather than parse
Markdown back apart. The two must never disagree, so this reuses `report.py`'s
own helpers for anything that carries a judgement call — which windows count as
"worst", when the duration-weighting caveat applies, what an exclusion reason is
called in plain English — rather than recomputing them a second, divergent way.

Nothing here recomputes a score. Like `report.py`, this is a view over
`result.json`.
"""

from __future__ import annotations

from typing import Any

from .report import EXCLUSION_LABELS, _basis_phrase, _timecode


def _window_evidence(w: dict[str, Any]) -> dict[str, Any]:
    c = w.get("components", {})
    return {
        "start": w["start"],
        "end": w["end"],
        "timecode": f"{_timecode(w['start'])}–{_timecode(w['end'])}",
        "score": w["score"],
        "verbatim": c.get("verbatim"),
        "semantic": c.get("semantic"),
        "onscreen_text": w.get("ocr_sentence_text", ""),
        "narration_text": w.get("asr_text", ""),
        "n_labels_excluded": int(c.get("n_label_elements", 0)),
        "shared_words": (w.get("evidence") or {}).get("shared_content_words") or [],
        "restated_sentences": [
            r["sentence"] for r in (w.get("evidence") or {}).get("restated_sentences") or []
        ],
    }


def _window_breakdown(w: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to recompute one window's score by hand.

    Feeds the web UI's segment pop-up: the on-screen text (what was compared,
    and what was set aside as labels), the narration, every sentence tested by
    the gate with its coverage and chance level, and the raw / chance /
    above-chance values of both overlap measures.
    """
    c = w.get("components", {})
    ev = w.get("evidence") or {}
    labels = list(dict.fromkeys(
        e["text"].replace("\n", " ").strip()
        for e in w.get("ocr_elements", [])
        if e.get("kind") == "label" and e.get("text", "").strip()
    ))
    reason = w.get("exclusion_reason")
    return {
        "index": w["index"],
        "start": w["start"],
        "end": w["end"],
        "duration": w["end"] - w["start"],
        "timecode": f"{_timecode(w['start'])}\u2013{_timecode(w['end'])}",
        "qualifies": w.get("qualifies", False),
        "exclusion_reason": reason,
        "exclusion_label": EXCLUSION_LABELS.get(reason, reason) if reason else None,
        "score": w.get("score"),
        "gate": ev.get("gate"),
        "onscreen_compared": w.get("ocr_sentence_text", ""),
        "onscreen_all": w.get("ocr_text", ""),
        "labels_excluded": labels[:20],
        "n_labels_excluded": len(labels),
        "narration": w.get("asr_text", ""),
        "sentence_checks": ev.get("sentence_checks") or [],
        "null_windows": ev.get("null_windows"),
        "verbatim_raw": c.get("verbatim_raw"),
        "verbatim_chance": c.get("verbatim_chance"),
        "verbatim": c.get("verbatim"),
        "semantic_raw": c.get("semantic_raw"),
        "semantic_chance": c.get("semantic_chance"),
        "semantic": c.get("semantic"),
        "onscreen_sentence_words": c.get("ocr_sentence_word_count"),
        "narration_words": c.get("asr_word_count"),
    }


def _worst_segments(payload: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    scored = [s for s in payload["segments"] if s["score"] is not None]
    # Mirrors report.py: a floor-scoring video has nothing to single out as
    # "worst" — every segment was assessed and found empty.
    if not scored or max(s["score"] for s in scored) == 0.0:
        return []

    worst = sorted(scored, key=lambda s: s["score"], reverse=True)[:top_n]
    by_index = {w["index"]: w for w in payload["result"]["windows"]}

    out = []
    for s in worst:
        qualifying = [
            by_index[i]
            for i in s.get("window_indices", [])
            if i in by_index and by_index[i].get("qualifies")
        ]
        qualifying.sort(key=lambda w: w["score"] or 0, reverse=True)
        out.append(
            {
                "segment_index": s["index"],
                "timecode": s["timecode"],
                "score": s["score"],
                "windows": [_window_evidence(w) for w in qualifying],
            }
        )
    return out


def _exclusions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for w in payload["result"]["windows"]:
        if not w.get("qualifies"):
            reason = w.get("exclusion_reason") or "unrecorded"
            counts[reason] = counts.get(reason, 0) + 1
    return [
        {"reason": reason, "label": EXCLUSION_LABELS.get(reason, reason), "count": n}
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def _coherence_window(w: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to recompute one window's coherence score by hand."""
    ev = w.get("evidence") or {}
    music = ev.get("music") or {}
    off = ev.get("off_topic") or {}
    c = w.get("components", {})
    return {
        "index": w["index"],
        "start": w["start"],
        "end": w["end"],
        "duration": w["end"] - w["start"],
        "timecode": f"{_timecode(w['start'])}\u2013{_timecode(w['end'])}",
        "qualifies": w.get("qualifies", False),
        "score": w.get("score"),
        "narration": ev.get("narration", ""),
        "background_music": c.get("background_music", 0.0),
        "off_topic": c.get("off_topic", 0.0),
        "narrated_s": music.get("narrated_s", 0.0),
        "music_under_speech_s": music.get("music_under_speech_s", 0.0),
        "music_without_speech_s": music.get("music_without_speech_s", 0.0),
        "spoken_at": music.get("spoken_at", []),
        "music_under_speech_at": music.get("music_under_speech_at", []),
        "music_without_speech_at": music.get("music_without_speech_at", []),
        "heard": music.get("heard", []),
        "music_to_speech_db": music.get("music_to_speech_db", []),
        "off_topic_check": off,
    }


def build_coherence_data(coh: dict[str, Any] | None) -> dict[str, Any] | None:
    """The coherence block for the UI, or None if the run did not score it."""
    if not coh:
        return None
    res = coh["result"]
    params = res["params"]
    by_index = {w["index"]: w for w in res.get("windows", [])}
    levels = next(
        ((w.get("evidence") or {}).get("off_topic") for w in res.get("windows", [])
         if ((w.get("evidence") or {}).get("off_topic") or {}).get("typical") is not None),
        None,
    ) or {}
    return {
        "score": res["score"],
        "status": res["status"],
        "na_reason": res.get("na_reason"),
        "warnings": res.get("warnings", []),
        "parts": coh.get("parts"),
        "alternative": coh.get("alternative"),
        "music_loudness": coh.get("music_loudness"),
        "assessed_score": coh.get("assessed_score"),
        "coverage": coh.get("coverage"),
        "n_windows_total": res["n_windows_total"],
        "n_windows_qualifying": res["n_windows_qualifying"],
        "off_topic_levels": {
            "typical": levels.get("typical"),
            "unrelated": levels.get("unrelated"),
            "cut": levels.get("cut"),
        },
        "windows": [_coherence_window(w) for w in res.get("windows", [])],
        "segments": [
            {
                "index": s["index"],
                "timecode": s["timecode"],
                "start": s["start"],
                "end": s["end"],
                "score": s["score"],
                "parts": s.get("parts") or {},
                "n_windows": s["n_windows"],
                "n_windows_qualifying": s["n_windows_qualifying"],
                "windows": [
                    _coherence_window(by_index[i])
                    for i in s.get("window_indices", []) if i in by_index
                ],
            }
            for s in coh["segments"]
        ],
        "methodology": params,
    }


def build_report_data(payload: dict[str, Any], *, top_n: int = 3) -> dict[str, Any]:
    """The data behind one video's report, as JSON-safe primitives."""
    res = payload["result"]
    rel = payload.get("ocr_recognition_reliability", {})
    cfg = payload.get("config", {})
    params = res["params"]
    windows_by_index = {w["index"]: w for w in res.get("windows", [])}

    return {
        "video_id": payload["video_id"],
        "video": payload["video"],
        "duration": payload["duration"],
        "score": res["score"],
        "assessed_score": payload.get("assessed_score"),
        "coverage": payload.get("coverage"),
        "aggregation": payload.get("aggregation"),
        "status": res["status"],
        "na_reason": res.get("na_reason"),
        "n_windows_total": res["n_windows_total"],
        "n_windows_qualifying": res["n_windows_qualifying"],
        "warnings": res.get("warnings", []),
        "reliability": {
            **rel,
            "basis_phrase": _basis_phrase(rel),
        },
        "segments": [
            {
                "windows": [
                    _window_breakdown(windows_by_index[i])
                    for i in s.get("window_indices", [])
                    if i in windows_by_index
                ],
                "index": s["index"],
                "timecode": s["timecode"],
                "start": s["start"],
                "end": s["end"],
                "score": s["score"],
                "verbatim": s.get("verbatim"),
                "semantic": s.get("semantic"),
                "n_windows": s["n_windows"],
                "n_windows_qualifying": s["n_windows_qualifying"],
                "assessed_score": s.get("assessed_score"),
                "coverage": s.get("coverage"),
                "measured_s": s.get("measured_s"),
            }
            for s in payload["segments"]
        ],
        "worst_segments": _worst_segments(payload, top_n),
        "coherence": build_coherence_data(payload.get("coherence")),
        "exclusions": _exclusions(payload),
        "methodology": {
            "weight_verbatim": params["weights"]["verbatim"],
            "weight_semantic": params["weights"]["semantic"],
            "min_onscreen_words": params["min_onscreen_words"],
            "min_narration_words": params["min_narration_words"],
            "embedding_model": params["embedding_model"],
            "window_size_s": cfg.get("windowing", {}).get("size_s"),
            "segment_length_s": payload["segment_length_s"],
            "min_sentence_coverage": (params.get("chance_correction") or {}).get(
                "min_sentence_coverage"),
            "min_shared_content_words": (params.get("chance_correction") or {}).get(
                "min_shared_content_words"),
            "null_exclusion_s": (params.get("chance_correction") or {}).get("null_exclusion_s"),
            "alignment_tolerance_s": cfg.get("alignment", {}).get("tolerance_s"),
            "frame_fps": cfg.get("media", {}).get("frame_fps"),
            "asr_backend": cfg.get("asr", {}).get("backend"),
            "ocr_backend": cfg.get("ocr", {}).get("backend"),
        },
    }
