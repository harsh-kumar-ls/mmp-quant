"""Export frames for the GPU cross-check of the decorative-picture detector.

    .venv/bin/python tools/export_frames_for_vlm.py      # -> build/vlm_frames.zip

13 Nov 2026 (user's GPU session on AIKosh, A100 slice with 5 GB): a small
vision-language model labels the pictures in a sample of frames as related /
unrelated to the lesson -- an independent check of the Mac-side detector, NOT
part of the score (scoring must run on the Mac for live uploads).

Sample: from each video's distinct frames (runs/<id>/frames, one per OCR
block), the first in every 5-second slot -- about one frame per 5 s. Each frame
carries the narration spoken within +/-5 s and the video's topic (its first
60 narrated words), so "related" can be judged against what is being taught.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "vlm_frames.zip"
SLOT_S = 5.0
CONTEXT_S = 5.0
MAX_WIDTH = 768


def _words(transcript: dict) -> list[tuple[float, float, str]]:
    out = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            out.append((float(w["start"]), float(w["end"]), str(w["text"]).strip()))
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, n_videos = [], 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        for run in sorted(p for p in (ROOT / "runs").iterdir()
                          if p.is_dir() and not p.name.startswith("_")):
            ocr, tr, frames = run / "ocr.json", run / "transcript.json", run / "frames"
            if not (ocr.exists() and tr.exists() and frames.is_dir()):
                continue
            blocks = json.loads(ocr.read_text())["blocks"]
            files = sorted(frames.glob("frame_*.png"))
            if len(files) != len(blocks):
                print(f"skip {run.name}: {len(files)} frames vs {len(blocks)} blocks")
                continue
            words = _words(json.loads(tr.read_text()))
            topic = " ".join(w for _, _, w in words[:60])
            n_videos += 1
            seen_slots: set[int] = set()
            for i, (b, f) in enumerate(zip(blocks, files, strict=True)):
                slot = int(float(b["start"]) // SLOT_S)
                if slot in seen_slots:
                    continue
                seen_slots.add(slot)
                t = float(b["start"])
                narration = " ".join(w for s, e, w in words
                                     if e > t - CONTEXT_S and s < t + CONTEXT_S)
                img = Image.open(f).convert("RGB")
                if img.width > MAX_WIDTH:
                    img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=82)
                name = f"frames/{run.name}/{i:05d}.jpg"
                z.writestr(name, buf.getvalue())
                rows.append({"video_id": run.name, "block": i, "time_s": round(t, 1),
                             "image": name, "topic": topic, "narration": narration})
        csv_buf = io.StringIO()
        w = csv.DictWriter(csv_buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        z.writestr("manifest.csv", csv_buf.getvalue())
    print(f"{OUT}: {len(rows)} frames from {n_videos} videos, "
          f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
