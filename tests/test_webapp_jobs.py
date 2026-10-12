"""JobManager's state machine, with `run_pipeline` stubbed out.

The real pipeline runs ASR/OCR child processes and takes minutes; these tests
replace it with a fast fake so the worker thread's bookkeeping — stage
transitions, one-at-a-time ordering, failure capture — is verified without
paying for a real video. The full real pipeline, run through this same job
manager via HTTP, was verified separately by hand (see PROGRESS.md): a real
upload reproduced the CLI's score for the same video exactly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mmpquant.webapp import jobs as jobs_module
from mmpquant.webapp.jobs import JobManager


def _wait_for(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(autouse=True)
def _require_project_root():
    # Config.load is called fresh per job (see JobManager._run_worker) against
    # a literal "config.yaml", so these tests must run from the project root —
    # same assumption the CLI itself makes.
    assert Path("config.yaml").exists(), "run tests from the project root"


class TestHappyPath:
    def test_job_starts_queued_or_already_picked_up(self, tmp_path, monkeypatch):
        def fake_run_pipeline(video, outdir, cfg, *, no_resume=False, on_stage=None):
            time.sleep(0.2)
            return object()

        monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
        manager = JobManager(Path("config.yaml"))
        job = manager.submit(tmp_path / "video.mp4", tmp_path / "out")
        # The worker thread may win the race and flip this to "running" before
        # this assertion runs — both are valid pre-completion states.
        assert job.status in ("queued", "running")

    def test_stage_transitions_are_recorded_in_order(self, tmp_path, monkeypatch):
        seen = []

        def fake_run_pipeline(video, outdir, cfg, *, no_resume=False, on_stage=None):
            for stage in ("asr", "ocr", "aligning", "scoring", "writing_report"):
                seen.append(stage)
                if on_stage:
                    on_stage(stage)
            return object()

        monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
        manager = JobManager(Path("config.yaml"))
        job = manager.submit(tmp_path / "video.mp4", tmp_path / "out")

        assert _wait_for(lambda: manager.get(job.job_id).status == "done")
        final = manager.get(job.job_id)
        assert final.status == "done"
        assert final.stage == "done"
        assert seen == ["asr", "ocr", "aligning", "scoring", "writing_report"]

    def test_to_dict_reports_stage_index_for_a_stepper_ui(self, tmp_path, monkeypatch):
        def fake_run_pipeline(video, outdir, cfg, *, no_resume=False, on_stage=None):
            on_stage("scoring")
            time.sleep(0.2)
            return object()

        monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
        manager = JobManager(Path("config.yaml"))
        job = manager.submit(tmp_path / "video.mp4", tmp_path / "out")

        assert _wait_for(lambda: manager.get(job.job_id).stage == "scoring")
        d = manager.get(job.job_id).to_dict()
        assert d["stage_index"] == jobs_module.STAGES.index(
            next(s for s in jobs_module.STAGES if s["key"] == "scoring")
        )
        assert d["stage_label"] == "Scoring redundancy & coherence"


class TestFailureIsReportedNotSwallowed:
    def test_a_raising_pipeline_marks_the_job_failed_with_the_error_text(self, tmp_path, monkeypatch):
        def fake_run_pipeline(video, outdir, cfg, *, no_resume=False, on_stage=None):
            on_stage("asr")
            raise RuntimeError("ffprobe exploded")

        monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
        manager = JobManager(Path("config.yaml"))
        job = manager.submit(tmp_path / "video.mp4", tmp_path / "out")

        assert _wait_for(lambda: manager.get(job.job_id).status == "failed")
        final = manager.get(job.job_id)
        assert final.status == "failed"
        assert "ffprobe exploded" in final.error

    def test_a_failed_job_does_not_stop_the_worker_from_taking_the_next_one(self, tmp_path, monkeypatch):
        calls = []

        def fake_run_pipeline(video, outdir, cfg, *, no_resume=False, on_stage=None):
            calls.append(video)
            if len(calls) == 1:
                raise RuntimeError("first video is corrupt")
            return object()

        monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
        manager = JobManager(Path("config.yaml"))
        job1 = manager.submit(tmp_path / "bad.mp4", tmp_path / "out1")
        job2 = manager.submit(tmp_path / "good.mp4", tmp_path / "out2")

        assert _wait_for(lambda: manager.get(job2.job_id).status == "done")
        assert manager.get(job1.job_id).status == "failed"
        assert manager.get(job2.job_id).status == "done"


class TestUnknownJob:
    def test_get_returns_none_for_an_unknown_id(self):
        manager = JobManager(Path("config.yaml"))
        assert manager.get("does-not-exist") is None
