"""Frames for the v5 GPU check: picture frames with every OCR text line greyed out.

    .venv/bin/python tools/export_masked_picture_frames.py   # -> build/vlm_frames_masked.zip

14 Nov 2026: the v4 forced choice (real narration vs a decoy from ANOTHER video)
scored 88.8% but matched all 4 of the user's extraneous pictures to their real
narration -- it tested the video's topic, partly by reading on-screen text (the
GDPR course title over a stock photo). v5 masks the text (OCR line boxes, from
ocr.json) so the model judges the picture alone, and uses decoys from the SAME
video. Only the 250 frames the v3 run said hold a picture (>= 8 narrated words).
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "vlm_frames_masked.zip"
MAX_WIDTH = 768


def main() -> None:
    src = zipfile.ZipFile(ROOT / "build" / "vlm_frames.zip")
    man = list(csv.DictReader(io.StringIO(src.read("manifest.csv").decode())))
    with open(ROOT / "data/vlm/vlm_labels_v3_2026-11-13.csv") as fh:
        pics = {(r["video_id"], int(r["block"])) for r in csv.DictReader(fh)
                if r["picture"] == "yes"}
    rows = [r for r in man if (r["video_id"], int(r["block"])) in pics
            and len(r["narration"].split()) >= 8]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            run = ROOT / "runs" / r["video_id"]
            i = int(r["block"])
            block = json.loads((run / "ocr.json").read_text())["blocks"][i]
            files = sorted((run / "frames").glob("frame_*.png"))
            img = Image.open(files[i]).convert("RGB")
            d = ImageDraw.Draw(img)
            for line in block.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(128, 128, 128))
            if img.width > MAX_WIDTH:
                img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            z.writestr(r["image"], buf.getvalue())
        # all narrations (for same-video decoys) + which frames to test
        z.writestr("manifest.csv", src.read("manifest.csv"))
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        z.writestr("test_frames.csv", out.getvalue())
    print(f"{OUT}: {len(rows)} masked picture frames, {OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
