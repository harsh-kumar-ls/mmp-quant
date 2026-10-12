"""The upload/report HTTP layer.

`job_manager.submit` is monkeypatched to a stub everywhere here — these tests
exercise the HTTP surface (filename safety, status codes, JSON shape), not the
scoring pipeline, which is a real subprocess-driven, multi-minute job covered
elsewhere (unit-tested in test_webapp_jobs.py; the real end-to-end path was
verified by hand — see PROGRESS.md).
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from mmpquant.webapp import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "VIDEOS_DIR", tmp_path / "videos")
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    return TestClient(server.app)


class TestSafeStem:
    def test_strips_directory_components(self):
        assert server._safe_stem("../../etc/passwd") == "passwd"

    def test_strips_unsafe_characters(self):
        assert server._safe_stem("my video (final)!.mp4") == "my_video_final"

    def test_falls_back_to_a_placeholder_for_an_empty_result(self):
        assert server._safe_stem("!!!.mp4") == "video"


class TestUniqueVideoPath:
    def test_first_upload_keeps_its_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "VIDEOS_DIR", tmp_path / "videos")
        monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
        path = server._unique_video_path("lecture", ".mp4")
        assert path.name == "lecture.mp4"

    def test_a_colliding_video_file_gets_suffixed_not_overwritten(self, tmp_path, monkeypatch):
        videos = tmp_path / "videos"
        videos.mkdir()
        (videos / "lecture.mp4").write_bytes(b"existing video - must not be clobbered")
        monkeypatch.setattr(server, "VIDEOS_DIR", videos)
        monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")

        path = server._unique_video_path("lecture", ".mp4")
        assert path.name == "lecture-2.mp4"
        assert (videos / "lecture.mp4").read_bytes() == b"existing video - must not be clobbered"

    def test_a_colliding_run_directory_also_forces_a_new_name(self, tmp_path, monkeypatch):
        # Different upload, same stem, no video file collision this time — but an
        # existing run directory would still be reused (and clobbered) without
        # this check, mixing a new video's artifacts into an old report.
        runs = tmp_path / "runs"
        (runs / "lecture").mkdir(parents=True)
        monkeypatch.setattr(server, "VIDEOS_DIR", tmp_path / "videos")
        monkeypatch.setattr(server, "RUNS_DIR", runs)

        path = server._unique_video_path("lecture", ".mp4")
        assert path.name == "lecture-2.mp4"


class TestUploadEndpoint:
    def test_rejects_an_unsupported_extension(self, client):
        res = client.post("/api/videos", files={"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")})
        assert res.status_code == 400

    def test_rejects_an_empty_file(self, client):
        res = client.post("/api/videos", files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")})
        assert res.status_code == 400

    def test_accepted_upload_is_queued_and_saved_to_disk(self, client, tmp_path, monkeypatch):
        submitted = {}

        class FakeJob:
            job_id = "abc123"

            def to_dict(self):
                return {"job_id": "abc123", "video_id": "clip", "status": "queued"}

        def fake_submit(video_path, outdir):
            submitted["video_path"] = video_path
            submitted["outdir"] = outdir
            return FakeJob()

        monkeypatch.setattr(server.job_manager, "submit", fake_submit)

        res = client.post("/api/videos", files={"file": ("clip.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")})
        assert res.status_code == 202
        assert res.json()["job_id"] == "abc123"
        assert submitted["video_path"].exists()
        assert submitted["video_path"].read_bytes() == b"fake video bytes"


class TestJobEndpoints:
    def test_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/does-not-exist").status_code == 404

    def test_report_for_a_job_not_yet_done_is_409(self, client, monkeypatch):
        class FakeJob:
            status = "running"

        monkeypatch.setattr(server.job_manager, "get", lambda job_id: FakeJob())
        res = client.get("/api/jobs/whatever/report")
        assert res.status_code == 409


class TestVideoEndpoints:
    def test_list_videos_is_empty_when_runs_dir_is_absent(self, client):
        assert client.get("/api/videos").json() == []

    def test_list_videos_reads_every_result_json(self, client, tmp_path, monkeypatch):
        runs = tmp_path / "runs2"
        monkeypatch.setattr(server, "RUNS_DIR", runs)
        for vid, score in [("a", 12.5), ("b", None)]:
            d = runs / vid
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({
                "video_id": vid, "duration": 60.0, "segments": [],
                "result": {"score": score, "status": "OK" if score is not None else "N/A"},
            }))
        videos = client.get("/api/videos").json()
        assert {v["video_id"] for v in videos} == {"a", "b"}

    def test_report_404s_when_no_result_json_exists(self, client):
        assert client.get("/api/videos/nonexistent/report").status_code == 404

    def test_report_matches_build_report_data(self, client, tmp_path, monkeypatch):
        runs = tmp_path / "runs3"
        d = runs / "demo"
        d.mkdir(parents=True)
        payload = _minimal_payload()
        (d / "result.json").write_text(json.dumps(payload))
        monkeypatch.setattr(server, "RUNS_DIR", runs)

        res = client.get("/api/videos/demo/report")
        assert res.status_code == 200
        assert res.json()["score"] == 55.0


class TestStagesEndpoint:
    def test_stages_matches_the_jobs_module_source_of_truth(self, client):
        from mmpquant.webapp.jobs import STAGES

        assert client.get("/api/stages").json() == STAGES


def _minimal_payload() -> dict:
    return {
        "video": "data/videos/demo.mp4", "video_id": "demo", "duration": 60.0,
        "segment_length_s": 30.0,
        "ocr_recognition_reliability": {"reliability": "ok"},
        "segments": [],
        "result": {
            "principle": "redundancy", "score": 55.0, "status": "OK", "na_reason": None,
            "n_windows_total": 1, "n_windows_qualifying": 1, "warnings": [],
            "params": {
                "weights": {"verbatim": 0.6, "semantic": 0.4},
                "min_onscreen_words": 5, "min_narration_words": 1,
                "embedding_model": "x",
            },
            "windows": [],
        },
        "config": {}, "config_path": "config.yaml",
    }


class TestStaticFilesAreRevalidated:
    """08 Oct 2026: without Cache-Control the browser kept a stale app.js and
    the updated UI appeared only after a hard refresh."""

    @pytest.mark.parametrize("path", ["/", "/app.js", "/styles.css"])
    def test_static_responses_carry_no_cache(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
