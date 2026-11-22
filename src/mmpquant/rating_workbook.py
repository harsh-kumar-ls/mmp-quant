"""The rating workbook sent to the professor (and any second rater).

One Excel file per rater instead of one CSV per video: a "Read me" sheet, a
"Ratings" sheet (every 30 s segment of every video, one row each), and a
"Video overall" sheet (one row per video). Both granularities are collected
because which one the professor prefers is still an open question; the
validation harness reads either.

It carries NO system output -- a rater shown the machine's score anchors to it,
and the correlation would then measure agreement with an anchor. Segment
boundaries come from the same `segments.length_s` the scorer uses, so rows join
to system segments with no reconciliation.

Each rater fills their own copy and saves it as ratings_<name>.xlsx; the
validation harness reads a folder of these, one rater per file.

12 Nov 2026 (after the professor's meeting): scale 1-100 (the professor's
choice), and the 'Video overall' sheet also asks for the three parts of
coherence the professor will rate -- background music and sounds, off-topic
narration, extraneous on-screen material -- named `<part>_score` so the
harness can read each one like a principle (`--principle music_sounds`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .segments import segment_bounds

RATINGS_SHEET = "Ratings"
VIDEO_SHEET = "Video overall"
README_SHEET = "Read me"
PRINCIPLES = ("redundancy", "coherence")
# rated once per whole video (12 Nov 2026); see the Rating Guide for definitions
COHERENCE_PARTS = ("music_sounds", "offtopic_narration", "onscreen_extraneous")


def _timecode(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def build_workbook(
    videos: Sequence[tuple[str, float]],
    out: Path,
    *,
    segment_length_s: float,
    scale: tuple[int, int] = (1, 100),
) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    lo, hi = scale
    wb = Workbook()
    head = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="2A78D6")
    shade = PatternFill("solid", fgColor="EEF3FB")
    wrap = Alignment(wrap_text=True, vertical="top")

    readme = wb.active
    readme.title = README_SHEET
    lines = [
        "Rating workbook — MMP-Quant (redundancy and coherence in educational videos)",
        "",
        "Please read the one-page Rating Guide before starting.",
        "",
        f"1. Sheet '{RATINGS_SHEET}': one row per 30-second segment. Watch the segment, then give",
        f"   a redundancy score and a coherence score, each a whole number from {lo} to {hi}",
        f"   ({lo} = none at all, {hi} = the worst possible).",
        f"2. Sheet '{VIDEO_SHEET}': after watching a whole video, give an overall redundancy and",
        "   coherence score, and score the three parts of coherence separately:",
        "   music_sounds (background music or sounds under the narration),",
        "   offtopic_narration (interesting but irrelevant talk),",
        "   onscreen_extraneous (decorative pictures, large logos, banners).",
        "3. Leave a cell EMPTY if you did not rate it. Empty means 'not rated', never 0.",
        "4. Optional: in 'extraneous', write M (music/sounds), O (off-topic talk),",
        "   D (decorative pictures, logos, banners) for what you noticed in that segment.",
        "5. Please rate without looking at the system's scores, and without discussing",
        "   your ratings with the other rater until both of you have finished.",
        "6. Save your copy as ratings_<your name>.xlsx and send it back.",
    ]
    for i, text in enumerate(lines, 1):
        readme.cell(row=i, column=1, value=text)
    readme["A1"].font = Font(bold=True, size=13)
    readme.column_dimensions["A"].width = 100

    score_check = DataValidation(type="whole", operator="between", formula1=str(lo),
                                 formula2=str(hi), allow_blank=True, showErrorMessage=True,
                                 errorTitle="Score out of range",
                                 error=f"Please enter a whole number from {lo} to {hi}, "
                                       "or leave the cell empty.")

    ratings = wb.create_sheet(RATINGS_SHEET)
    cols = ["video_id", "segment_index", "start_s", "end_s", "timecode",
            "redundancy_score", "coherence_score", "extraneous", "notes"]
    widths = [26, 9, 8, 8, 12, 18, 18, 12, 50]
    ratings.append(cols)
    ratings.add_data_validation(score_check)
    row = 2
    for v, (vid, duration) in enumerate(videos):
        for i, (start, end) in enumerate(segment_bounds(duration, segment_length_s)):
            ratings.append([vid, i, round(start, 1), round(end, 1),
                            f"{_timecode(start)}-{_timecode(end)}", None, None, None, None])
            if v % 2:
                for c in range(1, len(cols) + 1):
                    ratings.cell(row=row, column=c).fill = shade
            row += 1
    score_check.add(f"F2:G{max(2, row - 1)}")

    overall = wb.create_sheet(VIDEO_SHEET)
    parts = [f"{p}_score" for p in COHERENCE_PARTS]
    overall.append(["video_id", "duration", "redundancy_score", "coherence_score",
                    *parts, "notes"])
    overall_check = DataValidation(type="whole", operator="between", formula1=str(lo),
                                   formula2=str(hi), allow_blank=True, showErrorMessage=True,
                                   error=f"Please enter a whole number from {lo} to {hi}.")
    overall.add_data_validation(overall_check)
    for vid, duration in videos:
        overall.append([vid, _timecode(duration), None, None, *[None] * len(parts), None])
    overall_check.add(f"C2:{chr(ord('D') + len(parts))}{len(videos) + 1}")

    for sheet, ws in ((ratings, widths), (overall, [26, 10, 18, 18, 18, 22, 22, 60])):
        for c, w in enumerate(ws, 1):
            sheet.column_dimensions[sheet.cell(row=1, column=c).column_letter].width = w
            cell = sheet.cell(row=1, column=c)
            cell.font, cell.fill = head, head_fill
        sheet.freeze_panes = "A2"
        for r in sheet.iter_rows(min_row=2):
            r[-1].alignment = wrap

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
