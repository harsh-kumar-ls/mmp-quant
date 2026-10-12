"""FastAPI app: upload a video, poll job progress, read the structured report.

Run with `mmpquant serve` (see cli.py) from the project root — paths below
(`data/videos`, the run output directory) are resolved relative to the current
directory, exactly like the CLI.

Every number this app shows came from `mmpquant.pipeline.run_pipeline`, the
same function `mmpquant score` calls, and lands in `runs/<video_id>/` same as a
CLI run. This file adds no scoring behaviour — only upload handling, a job
queue, and JSON views over `result.json`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..report_data import build_report_data
from .jobs import STAGES, JobManager

log = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")
VIDEOS_DIR = Path("data/videos")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024  # 3 GB — generous for a lecture clip,
# bounded so a bad upload can't fill the disk unattended.

app = FastAPI(title="MMP-Quant")

_cfg = Config.load(CONFIG_PATH)
RUNS_DIR = Path(str(_cfg.get("output.dir", "runs")))
job_manager = JobManager(CONFIG_PATH)


def _safe_stem(filename: str) -> str:
    """A filesystem- and video-id-safe stem, derived from the upload's name.

    Strips any directory component (never trust a client-supplied path) and
    anything outside a conservative character set, so the result is safe to use
    both as a path segment and as `video_id` throughout the rest of the
    pipeline, which already uses the bare filename stem for that purpose.
    """
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "video"


def _unique_video_path(stem: str, suffix: str) -> Path:
    """Never silently overwrite an existing video or its run directory."""
    candidate = VIDEOS_DIR / f"{stem}{suffix}"
    n = 2
    while candidate.exists() or (RUNS_DIR / candidate.stem).exists():
        candidate = VIDEOS_DIR / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


@app.post("/api/videos")
def upload_video(file: UploadFile) -> JSONResponse:
    """Save an uploaded video and queue it for scoring.

    Runs as a plain (non-async) function so FastAPI executes it in its worker
    thread pool — a multi-hundred-megabyte upload copies to disk without
    blocking the event loop that other clients' status polling depends on.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"unsupported file type '{suffix or '(none)'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    target = _unique_video_path(_safe_stem(file.filename or "video"), suffix)

    written = 0
    with target.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds the 3 GB upload limit")
            out.write(chunk)

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "uploaded file is empty")

    job = job_manager.submit(target, RUNS_DIR / target.stem)
    log.info("accepted upload %s -> %s (job %s)", file.filename, target, job.job_id)
    return JSONResponse(job.to_dict(), status_code=202)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str, top: int = 3) -> dict:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}, not done yet")
    return _load_report(job.outdir, top=top)


@app.get("/api/stages")
def get_stages() -> list[dict]:
    """The stage list the frontend renders as a stepper, so it stays in sync
    with `jobs.STAGES` without hardcoding a copy in JavaScript."""
    return STAGES


@app.get("/api/videos")
def list_videos() -> list[dict]:
    """Every video already scored (by the CLI or this app), most recent first.

    Lets the UI show the 24-video corpus immediately rather than an empty
    screen until someone uploads something new.
    """
    out = []
    if not RUNS_DIR.exists():
        return out
    for result_path in RUNS_DIR.glob("*/result.json"):
        try:
            payload = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        res = payload.get("result", {})
        out.append(
            {
                "video_id": payload.get("video_id", result_path.parent.name),
                "score": res.get("score"),
                "status": res.get("status"),
                "duration": payload.get("duration"),
                "n_segments": len(payload.get("segments", [])),
                "scored_at": result_path.stat().st_mtime,
            }
        )
    out.sort(key=lambda v: v["scored_at"], reverse=True)
    return out


@app.get("/api/videos/{video_id}/report")
def get_video_report(video_id: str, top: int = 3) -> dict:
    return _load_report(RUNS_DIR / video_id, top=top)


def _load_report(outdir: Path, *, top: int) -> dict:
    result_path = outdir / "result.json"
    if not result_path.exists():
        raise HTTPException(404, f"no result.json in {outdir}")
    payload = json.loads(result_path.read_text())
    return build_report_data(payload, top_n=top)


# Registered last: a static-file mount at "/" is a catch-all, so every API
# route above must be declared before this line or it would shadow them.
_STATIC_DIR = Path(__file__).parent / "static"


class _RevalidatingStaticFiles(StaticFiles):
    """Static files the browser must re-check on every load.

    Without a Cache-Control header browsers cache app.js heuristically, so an
    updated UI only appeared after a hard refresh (seen twice, 05 and 08 Oct).
    `no-cache` still lets the browser keep a copy; it just revalidates it
    (a cheap 304 when unchanged) instead of trusting a stale one.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", _RevalidatingStaticFiles(directory=_STATIC_DIR, html=True), name="static")
