"""Build the read-only copy of the website (build/static-site/) -- option 1 of
the user's "1 + 3" (13 Nov 2026): a free static Hugging Face Space with every
scored video's report, no server, no uploading. Uploading is on the live site
(the student's Mac, option 3).

    .venv/bin/python tools/build_static_site.py

The React UI is built with VITE_STATIC=1, so it reads pre-built JSON instead of
calling the API:
    data/videos.json          = GET /api/videos
    data/stages.json          = GET /api/stages
    data/reports/<id>.json    = GET /api/videos/<id>/report
Each JSON is produced by the same functions the server uses, so the copy shows
exactly what the live site shows (the Mac's numbers = the thesis numbers).

Never copied: videos, frames, audio, rating packages, ratings, private notes.
To be made public only after the professor's ratings are back (user, 12 Nov 2026).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "static-site"
SPACE_README = """---
title: MMP-Quant
emoji: 📼
colorFrom: yellow
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Mayer redundancy and coherence scores for educational videos
---

# MMP-Quant (read-only copy)

Reports for every video scored in a B.Tech research project on Mayer's
redundancy and coherence principles: the score timelines, the instructor score
(100 - (redundancy + coherence) / 2), and the on-screen text beside the
narration for every 10-second window. Scored on a Mac with Apple Vision OCR:
these are the thesis numbers. Uploading a new video is only on the live site.
"""


def main(out: Path = OUT) -> None:
    os.chdir(ROOT)  # the server module resolves runs/ and config.yaml from here
    from mmpquant.report_data import build_report_data
    from mmpquant.webapp import server
    from mmpquant.webapp.jobs import STAGES

    if out.exists():
        shutil.rmtree(out)
    env = {**os.environ, "VITE_STATIC": "1", "VITE_BASE": "./"}
    subprocess.run(["npx", "vite", "build", "--outDir", str(out), "--emptyOutDir"],
                   cwd=ROOT / "webapp-react", env=env, check=True)

    data = out / "data"
    (data / "reports").mkdir(parents=True)
    videos = server.list_videos()
    (data / "videos.json").write_text(json.dumps(videos))
    (data / "stages.json").write_text(json.dumps(STAGES))
    for v in videos:
        payload = json.loads((server.RUNS_DIR / v["video_id"] / "result.json").read_text())
        report = build_report_data(payload, top_n=3)
        (data / "reports" / f"{v['video_id']}.json").write_text(json.dumps(report))
    (out / "README.md").write_text(SPACE_README)
    # the "Mark your own video" button's target; tools/share_site.sh replaces it on
    # the Space while the live site runs (13 Nov 2026). Offline by default.
    (out / "live.json").write_text(json.dumps({"url": None}))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out}: {len(videos)} reports, {size:.1f} MB")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OUT)
