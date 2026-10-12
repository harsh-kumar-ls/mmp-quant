#!/usr/bin/env python3
"""Generate synthetic videos with KNOWN redundancy, for pipeline smoke-testing.

These are NOT validation data and produce NO research result. They exist to
answer one engineering question: does the pipeline plumbing work end to end —
does ASR hear what we said, does OCR read what we drew, does alignment put them
in the same window, and does the scorer separate the two cases?

Because the on-screen text and the narration are both authored here, the
expected *ordering* is known by construction:

  redundant.mp4  slide text == narration, spoken while the slide is up  -> HIGH
  clean.mp4      same slide text, narration about something unrelated   -> LOW

The absolute numbers mean nothing (synthetic TTS over a plain white slide is
nothing like a real lecture). Only `score(redundant) >> score(clean)` is a
meaningful signal, and only about the code, not about Mayer's principle.

Requires macOS (`say`) and ffmpeg. Usage:
    python tools/make_synthetic_fixtures.py data/fixtures
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
VOICE = "Samantha"
RATE = 165

SLIDES = [
    "Mitochondria generate ATP through oxidative phosphorylation",
    "The electron transport chain pumps protons across the inner membrane",
    "ATP synthase converts the proton gradient into chemical energy",
]

# Keywords only, spatially separated -- the case the supervisor made explicit on
# 07 Sep 2026: keywords, topics and labels are NOT redundancy. Even when the
# narrator says these exact words, the video must score 0. This is the fixture
# that would have caught the scattered-label defect.
KEYWORD_SLIDES = [
    "MITOCHONDRIA",
    "ATP SYNTHASE",
    "PROTON GRADIENT",
]

# Deliberately unrelated to the slides: different topic, no shared content words.
# This is the EASY negative case -- it separates on topic alone, so a score can
# pass it without measuring redundancy at all.
UNRELATED_NARRATION = [
    "Imagine you are holding a bicycle wheel and you spin it as fast as you can",
    "Now tilt the axle sideways and feel how strongly it resists your hand",
    "That stubborn push back is angular momentum trying to keep its direction",
]

# THE DISCRIMINATING CASE. Same topic as the slides, explains and elaborates,
# but duplicates no wording -- i.e. exactly what Mayer says a well-designed video
# should do. A metric that merely detects topic similarity will wrongly flag this
# as redundant, and topic similarity is the failure mode the 0.4-weighted
# embedding branch is most prone to. If `complementary` does not score
# distinctly below `redundant`, the score is measuring the wrong thing.
COMPLEMENTARY_NARRATION = [
    "Think of this organelle as a tiny power station that never once stops running",
    "Picture a crowd being pushed uphill until they build up pressure on one side",
    "When that pressure is released it spins a turbine and the cell banks the fuel",
]


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{proc.stderr[-1500:]}")


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def synth_narration(lines: list[str], workdir: Path, name: str) -> tuple[Path, list[float]]:
    """Speak each line to its own file, concatenate, and return segment durations."""
    workdir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    durations: list[float] = []

    for i, line in enumerate(lines):
        aiff = workdir / f"{name}_{i}.aiff"
        run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), line])
        wav = workdir / f"{name}_{i}.wav"
        run(["ffmpeg", "-nostdin", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)])
        parts.append(wav)
        durations.append(duration_of(wav))

    listing = workdir / f"{name}_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    narration = workdir / f"{name}.wav"
    run([
        "ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", str(narration),
    ])
    return narration, durations


def render_keyword_slide(text: str, path: Path, size: tuple[int, int] = (1280, 720)) -> None:
    """One keyword per slide, plus two unrelated labels placed far apart.

    The spatial separation is the point: three short labels scattered across the
    frame must be read as three labels, never summed into one phrase.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT, 44)
    small = ImageFont.truetype(FONT, 34)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    w = draw.textbbox((0, 0), text, font=font)[2]
    draw.text(((size[0] - w) // 2, 120), text, fill="black", font=font)
    draw.text((80, 560), "FIG 1", fill="black", font=small)
    draw.text((size[0] - 320, 560), "STEP 2", fill="black", font=small)
    img.save(path)


def render_slide(text: str, path: Path, size: tuple[int, int] = (1280, 720)) -> None:
    """Render one slide as a PNG with Pillow.

    Pillow rather than ffmpeg's `drawtext` because Homebrew's ffmpeg 9 build
    ships without libfreetype, so the drawtext filter is unavailable. Rendering
    here is also more controllable, and the wrapped multi-line layout usefully
    exercises the OCR reading-order sort.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT, 40)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    lines = textwrap.wrap(text, width=42)
    line_height = 56
    y = (size[1] - line_height * len(lines)) // 2
    for line in lines:
        width = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((size[0] - width) // 2, y), line, fill="black", font=font)
        y += line_height

    img.save(path)


def build_video(narration: Path, durations: list[float], out: Path, workdir: Path,
                slides: list[str] | None = None, keyword_style: bool = False) -> None:
    """White slides showing SLIDES[i] for exactly the span narration line i occupies.

    Timing each slide to its audio segment is the whole point: it puts the
    on-screen text and its matching narration inside the same window, which is
    what the alignment layer is supposed to detect.
    """
    slide_paths: list[Path] = []
    for i, text in enumerate(slides if slides is not None else SLIDES):
        path = workdir / f"{out.stem}_slide_{i}.png"
        if keyword_style:
            render_keyword_slide(text, path)
        else:
            render_slide(text, path)
        slide_paths.append(path)

    # concat demuxer: the final entry must be repeated for its duration to apply
    entries = "".join(
        f"file '{p.resolve()}'\nduration {d:.3f}\n"
        for p, d in zip(slide_paths, durations)
    )
    entries += f"file '{slide_paths[-1].resolve()}'\n"
    listing = workdir / f"{out.stem}_slides.txt"
    listing.write_text(entries)

    run([
        "ffmpeg", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-i", str(narration),
        "-r", "25", "-fps_mode", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-shortest",
        str(out),
    ])


def main(outdir: Path) -> int:
    if sys.platform != "darwin":
        raise SystemExit("this fixture generator needs macOS `say`")
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    cases = {
        # The narrator reads the slide verbatim -> expect HIGH.
        "redundant": SLIDES,
        # Same slides, unrelated topic -> expect LOW (easy case).
        "clean": UNRELATED_NARRATION,
        # Same slides, same topic, no duplicated wording -> expect LOW.
        # This is the one that actually tests the metric.
        "complementary": COMPLEMENTARY_NARRATION,
    }
    for name, lines in cases.items():
        narration, durations = synth_narration(lines, work, name)
        build_video(narration, durations, outdir / f"{name}.mp4", work)
        print(f"wrote {outdir / f'{name}.mp4'}")

    # Keywords on screen, narrator saying those exact keywords. Expected: 0.
    kw_narration = [
        "Mitochondria are the site of cellular respiration in the cell",
        "ATP synthase sits in the inner membrane and does the real work",
        "The proton gradient stores the energy that drives the whole process",
    ]
    narration, durations = synth_narration(kw_narration, work, "keywords_only")
    build_video(narration, durations, outdir / "keywords_only.mp4", work,
                slides=KEYWORD_SLIDES, keyword_style=True)
    print(f"wrote {outdir / 'keywords_only.mp4'}")

    print("SYNTHETIC SMOKE-TEST FIXTURES — not validation data, no research result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixtures")))
