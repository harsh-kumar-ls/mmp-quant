"""One-video-at-a-time background scoring, run through the same pipeline the
CLI uses.

Jobs run strictly sequentially through a single worker thread, deliberately —
see `tools/batch_score.sh`: one video's ASR step alone saturates 4-5 CPU cores
on this machine, so scoring two uploads at once would only make them contend
for the same cores rather than finish faster. A second upload while one is
running is queued, not started.

Job state lives in memory only. That is a real limitation — restarting the
server loses the queue and the status of a running job — acceptable for a
single-user local tool where a lost job just means re-uploading, and simpler
than the alternative of a job database for something with no concurrent users.
The *output* of a finished job is never at risk: it lands in `runs/<video_id>/`
exactly where the CLI would put it, so nothing is lost but the in-progress
status display.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import Config
from ..pipeline import run_pipeline

log = logging.getLogger(__name__)

Status = Literal["queued", "running", "done", "failed"]

# Stage keys `pipeline.run_pipeline` reports via `on_stage`, in the order they
# occur, plus the two states that live outside the pipeline call itself. The
# label is what the frontend prints; the description is why that stage can
# take minutes, since a silent multi-minute wait reads as a hang otherwise.
STAGES: list[dict[str, str]] = [
    {"key": "queued", "label": "Queued", "description": "waiting for the video ahead of this one"},
    {"key": "asr", "label": "Transcribing narration", "description": "faster-whisper, runs on CPU"},
    {"key": "ocr", "label": "Reading on-screen text", "description": "OCR on sampled frames"},
    {"key": "audio", "label": "Listening for background music", "description": "audio classifier, about a minute"},
    {"key": "aligning", "label": "Aligning text to speech", "description": "seconds"},
    {"key": "scoring", "label": "Scoring redundancy & coherence", "description": "seconds"},
    {"key": "writing_report", "label": "Writing report", "description": "seconds"},
    {"key": "done", "label": "Done", "description": ""},
]
_STAGE_ORDER = [s["key"] for s in STAGES]


@dataclass
class Job:
    job_id: str
    video_id: str
    video_path: Path
    outdir: Path
    status: Status = "queued"
    stage: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": next((s["label"] for s in STAGES if s["key"] == self.stage), self.stage),
            "stage_index": _STAGE_ORDER.index(self.stage) if self.stage in _STAGE_ORDER else -1,
            "n_stages": len(STAGES),
            "error": self.error,
            "outdir": str(self.outdir),
        }


class JobManager:
    """Owns the single worker thread. One instance per server process."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def submit(self, video_path: Path, outdir: Path) -> Job:
        job = Job(job_id=uuid.uuid4().hex, video_id=video_path.stem,
                  video_path=video_path, outdir=outdir)
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        log.info("queued job %s for %s", job.job_id, job.video_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def _run_worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            self._set(job_id, status="running", stage="asr", started_at=time.time())
            try:
                cfg = Config.load(self._config_path)
                run_pipeline(
                    job.video_path,
                    job.outdir,
                    cfg,
                    on_stage=lambda stage, jid=job_id: self._set(jid, stage=stage),
                )
                self._set(job_id, status="done", stage="done", finished_at=time.time())
                log.info("job %s (%s) finished", job_id, job.video_id)
            except Exception as exc:  # noqa: BLE001 — reported to the caller, not swallowed
                log.error("job %s (%s) failed:\n%s", job_id, job.video_id, traceback.format_exc())
                self._set(job_id, status="failed", error=str(exc), finished_at=time.time())
