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
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..instructor import instructor_block
from ..report_data import build_report_data
from .jobs import STAGES, JobManager

log = logging.getLogger(__name__)

# MMPQUANT_CONFIG lets a deployment use its own config (the Hugging Face Space
# uses config.yaml + deploy/hf-space/overrides.yaml: RapidOCR off the Mac).
CONFIG_PATH = Path(os.environ.get("MMPQUANT_CONFIG", "config.yaml"))
VIDEOS_DIR = Path("data/videos")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024  # 3 GB — generous for a lecture clip,
# bounded so a bad upload can't fill the disk unattended.

app = FastAPI(title="MMP-Quant")

_cfg = Config.load(CONFIG_PATH)
RUNS_DIR = Path(str(_cfg.get("output.dir", "runs")))
job_manager = JobManager(CONFIG_PATH)

# Shared mode (`mmpquant serve --shared`, 13 Nov 2026): the site is opened to the
# professor through a tunnel. Her uploads go to data/shared_uploads/ and
# runs/_shared/ -- listed in the library like any video, but never part of the
# corpus, so they cannot change `mmpquant regress` or validation (two uploads
# through the web UI joined the corpus by accident on 11 Nov). Nothing is hidden.
SHARED = os.environ.get("MMPQUANT_SHARED") == "1"
SHARED_VIDEOS_DIR = Path("data/shared_uploads")
SHARED_RUNS_SUBDIR = "_shared"


def _upload_dirs() -> tuple[Path, Path]:
    """Where an upload's video file and its run directory go."""
    if SHARED:
        return SHARED_VIDEOS_DIR, RUNS_DIR / SHARED_RUNS_SUBDIR
    return VIDEOS_DIR, RUNS_DIR


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
    videos_dir, runs_dir = _upload_dirs()
    candidate = videos_dir / f"{stem}{suffix}"
    n = 2
    while (candidate.exists() or (runs_dir / candidate.stem).exists()
           or (RUNS_DIR / candidate.stem).exists()):
        candidate = videos_dir / f"{stem}-{n}{suffix}"
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

    videos_dir, runs_dir = _upload_dirs()
    videos_dir.mkdir(parents=True, exist_ok=True)
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

    job = job_manager.submit(target, runs_dir / target.stem)
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


def _styles() -> dict[str, str]:
    """Presentation style per video (data/video_styles.csv), if present."""
    from ..selection import load_styles

    return load_styles(Path("data/video_styles.csv"))


@app.get("/api/videos")
def list_videos() -> list[dict]:
    """Every video already scored (by the CLI or this app), most recent first.

    Lets the UI show the 24-video corpus immediately rather than an empty
    screen until someone uploads something new.
    """
    out = []
    if not RUNS_DIR.exists():
        return out
    styles = _styles()
    results = list(RUNS_DIR.glob("*/result.json"))
    results += list((RUNS_DIR / SHARED_RUNS_SUBDIR).glob("*/result.json"))
    for result_path in results:
        try:
            payload = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        res = payload.get("result", {})
        out.append(
            {
                "video_id": payload.get("video_id", result_path.parent.name),
                "score": res.get("score"),
                # for the library's filters (React UI, 10 Nov 2026)
                "style": styles.get(payload.get("video_id", result_path.parent.name)),
                "coherence": ((payload.get("coherence") or {}).get("result") or {}).get("score"),
                "instructor": (instructor_block(payload) or {}).get("score"),
                "status": res.get("status"),
                "duration": payload.get("duration"),
                "n_segments": len(payload.get("segments", [])),
                "scored_at": result_path.stat().st_mtime,
            }
        )
    out.sort(key=lambda v: v["scored_at"], reverse=True)
    return out


PACKAGES_GLOB = "rating_package*/selection.csv"


def _current_package() -> Path | None:
    """The professor's package: the newest rating_package*/ not marked
    SUPERSEDED in its README (10 Nov 2026)."""
    for sel in sorted(Path(".").glob(PACKAGES_GLOB), reverse=True):
        readme = sel.parent / "README.txt"
        if readme.exists() and "SUPERSEDED" in readme.read_text().splitlines()[0]:
            continue
        return sel.parent
    return None


@app.get("/api/rating-set")
def rating_set() -> dict:
    """The 10 videos chosen for the professor's ratings, with their targets and
    system scores (10 Nov 2026). STUDENT ONLY: these scores must never reach a
    rater before the ratings are back -- the UI says so on the page."""
    import csv

    pkg = _current_package()
    if pkg is None:
        raise HTTPException(404, "no rating package (run `mmpquant select-for-rating`)")
    rows = []
    for r in csv.DictReader((pkg / "selection.csv").open()):
        def num(x):
            return None if x in ("", None) else float(x)
        current = None
        result = RUNS_DIR / r["video_id"] / "result.json"
        if result.exists():
            payload = json.loads(result.read_text())
            current = {"redundancy": payload.get("result", {}).get("score"),
                       "coherence": ((payload.get("coherence") or {}).get("result") or {})
                       .get("score")}
        rows.append({
            "target": int(r["target"]), "video_id": r["video_id"], "style": r["style"],
            "duration_s": num(r["duration_s"]),
            "redundancy": num(r["redundancy"]), "distance": num(r["distance"]),
            "coherence": num(r["coherence"]), "music_loudness_db": num(r["music_loudness_db"]),
            "current": current,
        })
    rows.sort(key=lambda x: x["target"])
    return {"package": pkg.name, "rows": rows,
            "total_distance": round(sum(x["distance"] or 0 for x in rows), 1)}


@app.get("/api/videos/{video_id}/report")
def get_video_report(video_id: str, top: int = 3) -> dict:
    shared = RUNS_DIR / SHARED_RUNS_SUBDIR / video_id
    if not (RUNS_DIR / video_id / "result.json").exists() and (shared / "result.json").exists():
        return _load_report(shared, top=top)
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


# The React UI (10 Nov 2026, webapp-react/) is the main site once built
# (`npm run build`): it is served at / (and /react/, where its assets live).
# The original plain-JS UI stays available at /classic/ as the fallback. With
# no build present, / serves the original UI as before — no Node needed.
REACT_DIST = Path("webapp-react/dist")
app.mount("/classic", _RevalidatingStaticFiles(directory=_STATIC_DIR, html=True), name="classic")
if REACT_DIST.is_dir():
    app.mount("/react", _RevalidatingStaticFiles(directory=REACT_DIST, html=True), name="react")
    app.mount("/", _RevalidatingStaticFiles(directory=REACT_DIST, html=True), name="static")
else:
    app.mount("/", _RevalidatingStaticFiles(directory=_STATIC_DIR, html=True), name="static")
