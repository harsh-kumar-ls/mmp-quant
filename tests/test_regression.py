"""The regression loop's own checks must fail when they should -- a checker
that silently passes is worse than none. Synthetic payloads only."""

from __future__ import annotations

import yaml

from mmpquant import regression as rg


def payload(red=0.0, *, assessed=None, windows=None, coh=None, music_at=(), loud=None,
            parts=None, elements=None, checks=None):
    windows = windows or [{"start": 0.0, "score": red}]
    red_windows = [{"start": w["start"], "score": w["score"],
                    "ocr_elements": elements or [],
                    "evidence": {"sentence_checks": checks or []}} for w in windows]
    p = {"result": {"score": red, "windows": red_windows}, "assessed_score": assessed}
    if coh is not None:
        p["coherence"] = {
            "result": {"score": coh, "windows": [
                {"start": 0.0, "score": coh,
                 "evidence": {"music": {"music_under_speech_at": list(music_at)}}}]},
            "parts": parts or {"background_music": coh, "off_topic": 0.0},
            "music_loudness": {"median_db": loud},
        }
    return p


def run(check, results):
    return rg.check_one(check, results)


class TestEachCheckCanFail:
    def test_score(self):
        c = {"type": "score", "principle": "redundancy", "video": "a", "op": "==", "value": 0}
        assert run(c, {"a": payload(0.0)})[0] is True
        assert run(c, {"a": payload(12.5)})[0] is False

    def test_score_na_fails_rather_than_passing(self):
        c = {"type": "score", "principle": "redundancy", "video": "a", "op": "<=", "value": 5}
        assert run(c, {"a": payload(None)})[0] is False

    def test_missing_video_fails(self):
        c = {"type": "score", "principle": "redundancy", "video": "ghost", "op": "==", "value": 0}
        ok, detail = run(c, {"a": payload()})
        assert not ok and "no run" in detail

    def test_highest(self):
        c = {"type": "highest", "principle": "redundancy", "video": "a"}
        assert run(c, {"a": payload(70), "b": payload(30)})[0] is True
        assert run(c, {"a": payload(20), "b": payload(30)})[0] is False

    def test_window_mean(self):
        c = {"type": "window_mean", "principle": "redundancy", "video": "a",
             "from_s": 10, "to_s": 30, "op": ">=", "value": 80}
        ws = [{"start": 0.0, "score": 0.0}, {"start": 10.0, "score": 90.0},
              {"start": 20.0, "score": 60.0}]
        ok, detail = run(c, {"a": payload(windows=ws)})
        assert not ok and "75.00" in detail

    def test_never_sentence(self):
        c = {"type": "never_sentence", "video": "a", "text_contains": "aim"}
        label = [{"text": "Aim: provide language", "kind": "label"}]
        sentence = [{"text": "Aim: provide language", "kind": "sentence"}]
        assert run(c, {"a": payload(elements=label)})[0] is True
        assert run(c, {"a": payload(elements=sentence)})[0] is False

    def test_question_not_restated(self):
        c = {"type": "question_not_restated", "video": "a", "question": "What is x?"}
        passed = [{"sentence": "What is x?", "passed": True, "is_question": True}]
        assert run(c, {"a": payload(checks=passed)})[0] is False

    def test_text_never_counted(self):
        c = {"type": "text_never_counted", "video": "a", "text": "All areas get scaled by det(A)"}
        clean = payload(10.0)
        counted = payload(10.0)
        counted["result"]["windows"][0]["evidence"]["restated_sentences"] = [
            {"sentence": "All areas get scaled by det (A)"}]
        assert run(c, {"a": clean})[0] is True
        ok, detail = run(c, {"a": counted})
        assert not ok and "counted as redundancy" in detail

    def test_music_within_fails_outside_and_when_absent(self):
        c = {"type": "music_within", "video": "a", "from_s": 0, "to_s": 60, "tolerance_s": 5}
        assert run(c, {"a": payload(coh=10, music_at=[12, 30])})[0] is True
        assert run(c, {"a": payload(coh=10, music_at=[12, 90])})[0] is False
        assert run(c, {"a": payload(coh=0, music_at=[])})[0] is False

    def test_no_music_elsewhere_skips_listed_and_pending(self):
        c = {"type": "no_music_elsewhere", "music_videos": ["m"], "not_yet_listened": ["p"],
             "max_part": 1.0}
        res = {"m": payload(coh=90), "p": payload(coh=80), "q": payload(coh=0.5)}
        assert run(c, res)[0] is True
        res["q"] = payload(coh=12)
        assert run(c, res)[0] is False

    def test_louder(self):
        c = {"type": "louder", "video": "a", "than": ["b"]}
        assert run(c, {"a": payload(coh=1, loud=-8), "b": payload(coh=1, loud=-20)})[0] is True
        assert run(c, {"a": payload(coh=1, loud=-20), "b": payload(coh=1, loud=-8)})[0] is False

    def test_unknown_type_fails(self):
        assert run({"type": "vibes"}, {})[0] is False

    def test_a_broken_check_fails_loudly(self, tmp_path):
        f = tmp_path / "e.yaml"
        f.write_text(yaml.safe_dump({"checks": [{"id": "X", "type": "score", "video": "a"}]}))
        [r] = rg.check_expectations({"a": payload()}, f)
        assert not r.passed and "check error" in r.detail


class TestDiff:
    def test_unchanged_is_empty_and_rounding_is_ignored(self):
        a = rg.snapshot({"v": payload(10.0, coh=0.0)})
        b = rg.snapshot({"v": payload(10.004, coh=0.0)})
        assert rg.diff(a, b) == []

    def test_changes_name_the_windows_that_moved(self):
        before = rg.snapshot({"v": payload(10.0, windows=[{"start": 0, "score": 10.0},
                                                           {"start": 10, "score": 10.0}])})
        after = rg.snapshot({"v": payload(15.0, windows=[{"start": 0, "score": 10.0},
                                                          {"start": 10, "score": 20.0}])})
        [ch] = rg.diff(before, after)
        assert ch["kind"] == "redundancy" and ch["windows"] == [
            {"index": 1, "before": 10.0, "after": 20.0}]

    def test_added_and_removed_videos_are_reported(self):
        kinds = {c["kind"] for c in rg.diff(rg.snapshot({"a": payload()}),
                                            rg.snapshot({"b": payload()}))}
        assert kinds == {"added", "removed"}


def test_shipped_expectations_parse_and_name_their_source():
    spec = yaml.safe_load(rg.EXPECTATIONS.read_text())
    ids = [c["id"] for c in spec["checks"]]
    assert len(ids) == len(set(ids))
    for c in spec["checks"]:
        assert c.get("source"), f"{c['id']} must say where the expectation came from"
        assert c.get("status") in {"draft", "confirmed"}


def test_expectations_never_contain_manual_scores():
    # The professor's scores are for validation only; tuning to them here
    # would make the validation circular. (The file's header may say so.)
    checks = yaml.safe_dump(yaml.safe_load(rg.EXPECTATIONS.read_text())["checks"]).lower()
    assert "manual_score" not in checks and "professor" not in checks


def test_window_each_names_every_failing_window():
    c = {"type": "window_each", "principle": "redundancy", "video": "a",
         "from_s": 10, "to_s": 40, "op": ">=", "value": 90}
    ws = [{"start": 0.0, "score": 0.0}, {"start": 10.0, "score": 95.0},
          {"start": 20.0, "score": 61.0}, {"start": 30.0, "score": 99.0}]
    ok, detail = rg.check_one(c, {"a": payload(windows=ws)})
    assert not ok and "1 of 3 windows fail: [(20, 61.0)]" in detail
