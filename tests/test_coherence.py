"""Coherence scorer (11 Oct 2026): background music under speech, and
off-topic narration.

Synthetic fixtures only: a fake transcript with word timings, a fake sound
timeline, and a toy embedder whose "meaning" is which topic word-list a text
draws from -- so every expected value can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmpquant.config import Config, ConfigError
from mmpquant.extract.audio_events import AudioEvents, AudioFrame, frame_clips
from mmpquant.scoring.base import VideoContext
from mmpquant.scoring.coherence import (
    EXCL_NO_NARRATION,
    PART_MUSIC,
    PART_OFF_TOPIC,
    CoherenceScorer,
    load_reference_texts,
    part_scores,
)
from mmpquant.segments import aggregate_to_segments, aggregate_video
from mmpquant.types import AsrSegment, AsrWord, Transcript

from .conftest import make_window

TOPICS = {
    "physics": {"force", "mass", "energy", "velocity", "momentum", "newton",
                "acceleration", "motion", "inertia", "friction"},
    "cooking": {"onion", "garlic", "sauce", "oven", "bake", "flour", "butter",
                "simmer", "pepper", "salt"},
    "sport": {"match", "goal", "team", "penalty", "coach", "league", "score",
              "season", "player", "referee"},
}


class TopicEncoder:
    """A text's vector = how many of its words fall in each topic list, plus a
    small shared component (real embeddings of any two English texts are
    never orthogonal)."""

    def encode(self, texts):
        rows = []
        for t in texts:
            words = t.lower().split()
            v = [sum(w in vocab for w in words) for vocab in TOPICS.values()] + [0.3]
            v = np.asarray(v, dtype=float)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows)


REFERENCES = (
    "onion garlic sauce simmer salt pepper butter",
    "match goal team penalty coach referee season",
    "flour butter oven bake salt",
    "league player score season coach",
)

PHYSICS = "force mass energy velocity momentum newton acceleration motion inertia friction"
COOKING = "onion garlic sauce oven bake flour butter simmer pepper salt"


def transcript(spans):
    """spans: (start, end, text) -> one segment per span, words spread evenly."""
    segments = []
    for start, end, text in spans:
        toks = text.split()
        step = (end - start) / len(toks)
        words = tuple(
            AsrWord(start + i * step, start + (i + 1) * step, tok) for i, tok in enumerate(toks)
        )
        segments.append(AsrSegment(start, end, text, words))
    return Transcript(tuple(segments), "en", 0.99, max(e for _, e, _ in spans))


def events(n_seconds, music=lambda t: 0.0):
    frames = tuple(
        AudioFrame(float(t), float(t + 1), {"Speech": 0.8, "Music": music(t)},
                   top=(("Music", music(t)),) if music(t) else ())
        for t in range(n_seconds)
    )
    return AudioEvents(frames, 1.0, 3.0, "test-model", ("Speech", "Music"))


def windows(n, size=10.0):
    return [make_window("", "", index=i, start=i * size, end=(i + 1) * size) for i in range(n)]


def scorer(**kw):
    kw.setdefault("reference_texts", REFERENCES)
    return CoherenceScorer(TopicEncoder(), **kw)


def run(spans, n_windows, music=lambda t: 0.0, **kw):
    ctx = VideoContext(transcript=transcript(spans), audio_events=events(n_windows * 10, music))
    return scorer(**kw).score(windows(n_windows), context=ctx)


class TestBackgroundMusic:
    def test_music_under_all_speech_is_a_full_share(self):
        r = run([(0, 10, PHYSICS)], 1, music=lambda t: 0.6)
        ws = r.window_scores[0]
        assert ws.components[PART_MUSIC] == 1.0
        assert ws.score == pytest.approx(50.0)  # 100 × 0.5 × 1.0, on-topic

    def test_music_under_half_the_speech_is_half(self):
        r = run([(0, 10, PHYSICS)], 1, music=lambda t: 0.6 if t < 5 else 0.0)
        assert r.window_scores[0].components[PART_MUSIC] == pytest.approx(0.5)

    def test_music_with_nobody_speaking_does_not_count_but_is_recorded(self):
        # An intro jingle: music 0-5 s, narration 5-10 s with no music under it.
        r = run([(5, 10, PHYSICS)], 1, music=lambda t: 0.7 if t < 5 else 0.0)
        ws = r.window_scores[0]
        assert ws.components[PART_MUSIC] == 0.0
        assert ws.evidence["music"]["music_without_speech_s"] == pytest.approx(5.0)
        assert ws.evidence["music"]["music_without_speech_at"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_threshold_is_inclusive_and_configurable(self):
        at = run([(0, 10, PHYSICS)], 1, music=lambda t: 0.3, music_threshold=0.3)
        below = run([(0, 10, PHYSICS)], 1, music=lambda t: 0.29, music_threshold=0.3)
        assert at.window_scores[0].components[PART_MUSIC] == 1.0
        assert below.window_scores[0].components[PART_MUSIC] == 0.0

    def test_only_spoken_seconds_form_the_denominator(self):
        # Speech 0-4 s, music throughout: the share is of the 4 spoken s.
        r = run([(0, 4, "force mass energy velocity")], 1, music=lambda t: 0.6)
        ws = r.window_scores[0]
        assert ws.components["narrated_s"] == pytest.approx(4.0)
        assert ws.components[PART_MUSIC] == 1.0


class TestOffTopic:
    SPANS = [(i * 10, i * 10 + 9, PHYSICS) for i in range(5)] + [(50, 59, COOKING)]

    def test_a_cooking_aside_in_a_physics_video_is_off_topic(self):
        r = run(self.SPANS, 6)
        flags = [ws.components[PART_OFF_TOPIC] for ws in r.window_scores]
        assert flags == [0, 0, 0, 0, 0, 1.0]
        assert r.window_scores[5].score == pytest.approx(50.0)

    def test_evidence_shows_relevance_against_typical_and_unrelated(self):
        r = run(self.SPANS, 6)
        ev = r.window_scores[5].evidence[PART_OFF_TOPIC]
        assert ev["judged"] and ev["off_topic"]
        assert ev["unrelated"] < ev["cut"] < ev["typical"]
        assert ev["relevance"] < ev["cut"]
        # the cut is the midpoint at the default decision point
        assert ev["cut"] == pytest.approx((ev["unrelated"] + ev["typical"]) / 2, abs=1e-3)

    def test_a_consistently_on_topic_video_has_nothing_off_topic(self):
        r = run([(i * 10, i * 10 + 9, PHYSICS) for i in range(6)], 6)
        assert all(ws.components[PART_OFF_TOPIC] == 0 for ws in r.window_scores)
        assert r.score == 0.0

    def test_short_narration_is_not_judged(self):
        spans = self.SPANS[:5] + [(50, 52, "onion garlic")]  # 2 words < 8
        r = run(spans, 6)
        ev = r.window_scores[5].evidence[PART_OFF_TOPIC]
        assert ev["judged"] is False
        assert "2 spoken word" in ev["reason_not_judged"]
        assert r.window_scores[5].components[PART_OFF_TOPIC] == 0.0

    def test_too_few_judgeable_windows_warns_and_flags_nothing(self):
        r = run([(0, 9, PHYSICS), (10, 19, COOKING)], 2)
        assert all(ws.components[PART_OFF_TOPIC] == 0 for ws in r.window_scores)
        assert any("off-topic part not assessed" in w for w in r.warnings)

    def test_no_topic_closer_than_unrelated_text_is_not_assessed(self):
        # Every window draws on a different topic, so the narration is no more
        # like itself than like the reference paragraphs.
        spans = [(0, 9, PHYSICS), (10, 19, COOKING),
                 (20, 29, "match goal team penalty coach league score season player referee")]
        r = run(spans, 3, min_assessed_windows=3)
        assert all(ws.components[PART_OFF_TOPIC] == 0 for ws in r.window_scores)
        assert any("no closer to itself" in w for w in r.warnings)


class TestWindowAndVideo:
    def test_window_with_no_speech_does_not_qualify(self):
        r = run([(0, 9, PHYSICS)], 2)
        assert r.window_scores[1].qualifies is False
        assert r.window_scores[1].exclusion_reason == EXCL_NO_NARRATION

    def test_no_narration_anywhere_is_na_not_zero(self):
        ctx = VideoContext(transcript=Transcript((), None, None, 30.0), audio_events=events(30))
        r = scorer().score(windows(3), context=ctx)
        assert r.score is None and r.status == "N/A"
        assert "no narration" in r.na_reason

    def test_both_parts_combine_by_the_weights(self):
        spans = [(i * 10, i * 10 + 9, PHYSICS) for i in range(5)] + [(50, 59, COOKING)]
        r = run(spans, 6, music=lambda t: 0.6 if t >= 50 else 0.0,
                weight_music=0.25, weight_off_topic=0.75)
        assert r.window_scores[5].score == pytest.approx(100.0)  # music 1, off-topic 1
        assert r.window_scores[0].score == 0.0

    def test_parts_add_up_to_the_whole_video_score(self):
        spans = [(i * 10, i * 10 + 9, PHYSICS) for i in range(5)] + [(50, 59, COOKING)]
        r = run(spans, 6, music=lambda t: 0.6 if t < 10 else 0.0)
        agg = aggregate_video(r.window_scores)
        parts = part_scores(r.window_scores)
        assert agg.score == pytest.approx(0.5 * parts[PART_MUSIC] + 0.5 * parts[PART_OFF_TOPIC],
                                          abs=0.01)

    def test_segments_carry_both_parts(self):
        r = run([(0, 29, PHYSICS + " " + PHYSICS)], 3, music=lambda t: 0.6)
        seg = aggregate_to_segments(r.window_scores, 30.0, 30.0,
                                    part_keys=(PART_MUSIC, PART_OFF_TOPIC))[0]
        assert set(seg.parts) == {PART_MUSIC, PART_OFF_TOPIC}
        assert seg.verbatim is None and seg.semantic is None

    def test_score_requires_the_video_context(self):
        with pytest.raises(ValueError, match="VideoContext"):
            scorer().score(windows(1))


class TestReferenceTextsAndFraming:
    def test_package_reference_file_loads_without_comments(self):
        refs = load_reference_texts("unrelated_reference.txt")
        assert len(refs) >= 30
        assert not any(r.startswith("#") for r in refs)

    def test_reference_file_has_no_channel_boilerplate(self):
        # Deliberate: the baseline must not be built to catch one pattern.
        text = " ".join(load_reference_texts("unrelated_reference.txt")).lower()
        assert "subscribe" not in text and "sponsor" not in text

    def test_frames_are_one_hop_long_with_a_centred_clip(self):
        spans = frame_clips(n_samples=16000 * 3 + 800, sr=16000, hop_s=1.0, clip_s=3.0)
        assert [(a, b) for a, b, _, _ in spans[:3]] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        _, _, lo, hi = spans[1]
        assert (lo, hi) == (0, 48000)  # centred on 1.5 s
        assert spans[-1][:2] == (3.0, 3.05)  # partial final frame kept

    def test_audio_events_round_trip(self):
        ev = events(3, music=lambda t: 0.5)
        assert AudioEvents.from_dict(ev.to_dict()) == ev


class TestConfig:
    def _cfg(self, **coherence_overrides):
        cfg = Config.load("config.yaml")
        raw = dict(cfg.raw)
        raw["scoring"] = {**raw["scoring"],
                          "coherence": {**raw["scoring"]["coherence"], **coherence_overrides}}
        return Config(raw=raw)

    def test_shipped_config_is_valid(self):
        Config.load("config.yaml").validate()

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ConfigError, match="sum to 1.0"):
            self._cfg(weights={"background_music": 0.5, "off_topic": 0.6}).validate()

    def test_unknown_principle_is_rejected(self):
        cfg = Config.load("config.yaml")
        raw = {**cfg.raw, "scoring": {**cfg.raw["scoring"], "principles": ["signaling"]}}
        with pytest.raises(ConfigError, match="unknown principle"):
            Config(raw=raw).validate()


class TestReportAndUiData:
    """The Markdown report and the web UI's data must show the same coherence
    numbers the scorer produced, with their working."""

    def _payload(self, music=lambda t: 0.6 if t < 10 else 0.0):
        from mmpquant.pipeline import coherence_block

        from .test_report import payload

        spans = [(i * 10, i * 10 + 9, PHYSICS) for i in range(5)] + [(50, 59, COOKING)]
        r = run(spans, 6, music=music)
        segs = aggregate_to_segments(r.window_scores, 60.0, 30.0,
                                     part_keys=(PART_MUSIC, PART_OFF_TOPIC))
        p = payload()
        p["coherence"] = coherence_block(r, segs, video_label="demo")
        return p

    def test_block_parts_combine_to_the_reported_score(self):
        coh = self._payload()["coherence"]
        w = coh["result"]["params"]["weights"]
        assert coh["result"]["score"] == pytest.approx(
            w[PART_MUSIC] * coh["parts"][PART_MUSIC] + w[PART_OFF_TOPIC] * coh["parts"][PART_OFF_TOPIC],
            abs=0.01)

    def test_markdown_has_score_parts_evidence_and_method(self):
        from mmpquant.report import render_markdown

        md = render_markdown(self._payload())
        assert "## Coherence" in md
        assert "Background music under narration" in md
        assert "Music under narration at: 0:00–0:09" in md
        assert "| 0:50–1:00 |" in md  # the off-topic window, with its narration
        assert "music with nobody speaking is not counted" in md

    def test_markdown_omits_the_section_for_runs_without_coherence(self):
        from mmpquant.report import render_markdown

        from .test_report import payload

        assert "## Coherence" not in render_markdown(payload())

    def test_ui_data_carries_segments_windows_and_levels(self):
        from mmpquant.report_data import build_report_data

        c = build_report_data(self._payload())["coherence"]
        assert c["score"] is not None and set(c["parts"]) == {PART_MUSIC, PART_OFF_TOPIC}
        lv = c["off_topic_levels"]
        assert lv["unrelated"] < lv["cut"] < lv["typical"]
        w = c["segments"][1]["windows"][-1]  # 0:50-1:00, the cooking aside
        assert w["off_topic"] == 1.0 and w["off_topic_check"]["off_topic"] is True
        assert c["segments"][0]["windows"][0]["spoken_at"][:3] == [0.0, 1.0, 2.0]

    def test_ui_data_is_none_for_runs_without_coherence(self):
        from mmpquant.report_data import build_report_data

        from .test_report import payload

        assert build_report_data(payload())["coherence"] is None


class TestOffTopicKeptOutOfTheScore:
    """11 Oct 2026 (user decision): off-topic is computed and shown but has
    weight 0; the approved 0.5/0.5 is stored as the pre-registered alternative."""

    def _block(self):
        from mmpquant.pipeline import coherence_block

        spans = [(i * 10, i * 10 + 9, PHYSICS) for i in range(5)] + [(50, 59, COOKING)]
        r = run(spans, 6, music=lambda t: 0.6 if t < 10 else 0.0,
                weight_music=1.0, weight_off_topic=0.0)
        segs = aggregate_to_segments(r.window_scores, 60.0, 30.0,
                                     part_keys=(PART_MUSIC, PART_OFF_TOPIC))
        return coherence_block(r, segs, video_label="demo",
                               alternative_weights={PART_MUSIC: 0.5, PART_OFF_TOPIC: 0.5})

    def test_score_is_music_only_but_off_topic_is_still_measured(self):
        b = self._block()
        assert b["result"]["score"] == pytest.approx(b["parts"][PART_MUSIC], abs=0.01)
        assert b["parts"][PART_OFF_TOPIC] > 0  # the cooking aside is still found

    def test_alternative_is_the_equal_weighting_of_the_same_parts(self):
        b = self._block()
        expected = 0.5 * b["parts"][PART_MUSIC] + 0.5 * b["parts"][PART_OFF_TOPIC]
        assert b["alternative"]["score"] == pytest.approx(expected, abs=0.01)

    def test_report_labels_off_topic_experimental_and_shows_the_alternative(self):
        from mmpquant.report import render_markdown

        from .test_report import payload

        p = payload()
        p["coherence"] = self._block()
        md = render_markdown(p)
        assert "Off-topic narration (experimental — not in the score)" in md
        assert "Pre-registered alternative (0.5 / 0.5" in md

    def test_shipped_config_keeps_off_topic_out_and_registers_the_alternative(self):
        cfg = Config.load("config.yaml")
        assert cfg.get("scoring.coherence.weights") == {PART_MUSIC: 1.0, PART_OFF_TOPIC: 0.0}
        assert cfg.get("scoring.coherence.alternative_weights") == {PART_MUSIC: 0.5,
                                                                    PART_OFF_TOPIC: 0.5}


class TestMusicLoudness:
    """14 Oct 2026: music level relative to speech, shown as evidence only.
    The user heard test_7's music as louder than test6's / Test_11's."""

    def test_quietest_block_measures_the_music_between_words(self):
        from mmpquant.extract.audio_events import frame_levels

        sr = 16000
        t = np.arange(sr) / sr
        music = 0.05 * np.sin(2 * np.pi * 220 * t)          # throughout the second
        voice = np.where(t < 0.8, 0.5 * np.sin(2 * np.pi * 150 * t), 0.0)  # pause at 0.8-1.0 s
        level, floor = frame_levels((music + voice).astype(np.float32), sr, 0.0, 1.0)
        # the quietest 100 ms holds only the music: 0.05 amplitude sine -> RMS 0.0354 -> -29 dB
        assert floor == pytest.approx(20 * np.log10(0.05 / np.sqrt(2)), abs=0.3)
        assert level > floor + 10

    def test_digital_silence_is_clamped_not_minus_infinity(self):
        from mmpquant.extract.audio_events import MIN_DB, frame_levels

        assert frame_levels(np.zeros(16000, np.float32), 16000, 0.0, 1.0) == (MIN_DB, MIN_DB)

    def test_evidence_and_video_median_but_score_unchanged(self):
        from dataclasses import replace

        from mmpquant.scoring.coherence import music_loudness_db

        ev = events(10, music=lambda t: 0.6)
        ev = replace(ev, frames=tuple(replace(f, level_db=-20.0, floor_db=-26.0 - (f.start % 2))
                                      for f in ev.frames))
        ctx = VideoContext(transcript=transcript([(0, 10, PHYSICS)]), audio_events=ev)
        r = scorer().score(windows(1), context=ctx)
        diffs = r.window_scores[0].evidence["music"]["music_to_speech_db"]
        assert sorted(set(diffs)) == [-7.0, -6.0]
        assert music_loudness_db(r.window_scores) == {"median_db": -6.5, "n_seconds": 10}
        assert r.window_scores[0].components[PART_MUSIC] == 1.0  # loudness never enters the score

    def test_no_music_means_no_loudness(self):
        from mmpquant.scoring.coherence import music_loudness_db

        r = run([(0, 10, PHYSICS)], 1)
        assert music_loudness_db(r.window_scores) == {"median_db": None, "n_seconds": 0}
