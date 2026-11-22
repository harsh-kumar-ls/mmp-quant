"""Write the rater's Rating_Guide.docx (12 Nov 2026, after the professor's meeting).

Usage:
    .venv/bin/python tools/make_rating_guide.py OUT.docx --captions {count,exclude,pending}

The caption rule waits for the professor's reply to the Mayer (2021) evidence
(user's decision, 12 Nov 2026): `pending` writes a highlighted placeholder and
must never be sent. Everything else in the guide was agreed with the user on
12 Nov 2026 -- signals (labels, keywords, insights, aims, results) are not
redundancy; coherence has three parts rated once per video; scale 1-100;
instructor score = 100 - (R + C) / 2.

Needs the `package` extra (python-docx). Holds no system scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

CAPTION_DEFINITION = (
    "A caption is a full sentence that the narrator is saying, written across the "
    "bottom of the picture while it is spoken, like YouTube subtitles, but part of "
    "the video itself so the viewer cannot switch it off.")

CAPTIONS = {
    "count": ("Captions count as redundancy. " + CAPTION_DEFINITION + " Mayer (2021) states "
              "the redundancy principle for videos as “do not add captions that repeat "
              "the spoken words”."),
    "exclude": ("Captions do not count as redundancy (your decision). " + CAPTION_DEFINITION
                + " Please judge redundancy only from the other on-screen text."),
    "pending": "[CAPTION RULE: WAITING FOR THE PROFESSOR'S REPLY. DO NOT SEND THIS VERSION.]",
}

INK = RGBColor(0x1D, 0x1A, 0x16)


def _heading(doc, text, size=13):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(size), INK
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    return p


def _para(doc, text, *, bold_lead=None, highlight=False):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        r.bold = True
    p.paragraph_format.space_after = Pt(4)
    return p


def _bullets(doc, items):
    for lead, text in items:
        p = doc.add_paragraph(style="List Bullet")
        if lead:
            p.add_run(lead).bold = True
        p.add_run(text)
        p.paragraph_format.space_after = Pt(2)


def _cell_style(cell, *, fill=None):
    """Explicit borders (and header shading): a named table style is not drawn
    by every viewer the professor might open the guide in."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "9A9184")
        borders.append(el)
    tc_pr.append(borders)
    if fill:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        tc_pr.append(shd)


def _scale_table(doc, rows):
    t = doc.add_table(rows=1, cols=2)
    t.autofit = False
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Score", "What it looks like"
    for score, text in rows:
        cells = t.add_row().cells
        cells[0].text, cells[1].text = score, text
    for i, row in enumerate(t.rows):
        for cell, width in zip(row.cells, (Pt(50), Pt(430))):
            cell.width = width
            _cell_style(cell, fill="EFE8DC" if i == 0 else None)
            if i == 0:
                cell.paragraphs[0].runs[0].bold = True


def build(out: Path, captions: str) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Pt(54)

    title = doc.add_paragraph()
    r = title.add_run("Rating guide: redundancy and coherence in educational videos")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(16), INK
    _para(doc, "Version of 13 November 2026. Your ratings are the reference that an automatic "
               "scoring system will be compared with. Please rate from your own judgement; the "
               "system's scores are deliberately not shown anywhere in the package.")

    _heading(doc, "How to rate")
    _bullets(doc, [
        ("Sheet ‘Ratings’, one row per 30-second segment: ",
         ("watch the segment (replaying is fine), then give a redundancy score and a coherence "
         "score, each a whole number from 1 to 100.")),
        ("Sheet ‘Video overall’, one row per video: ",
         ("after the whole video, give an overall redundancy score and coherence score, and also "
         "score the three parts of coherence separately (music_sounds, offtopic_narration, "
         "onscreen_extraneous; defined below), each from 1 to 100.")),
        ("Scale: ", ("1 = none at all, 100 = the worst possible. Any whole number in between is "
                    "fine; the tables below give anchors at 1, 25, 50, 75 and 100.")),
        ("Empty cell = not rated, ", "never 0."),
        ("Optional: ", ("in the ‘extraneous’ column of a segment, write M (music or "
                       "sounds), O (off-topic talk) or D (decorative pictures, logos, banners) "
                       "for anything extraneous you noticed. The notes column is welcome for "
                       "any judgement call.")),
        ("Size: ", ("10 videos, about 29 minutes in total; 63 segments. One or two sittings "
                   "works well.")),
    ])

    _heading(doc, "Redundancy (Mayer)")
    _para(doc, "People learn less well when printed text on the screen repeats what the "
               "narrator is saying at the same moment, because they must read and listen to "
               "the same words at once. Rate how much of the time the on-screen text is being "
               "read aloud, word for word or closely restated.")
    _para(doc, "Counts as redundancy: ", bold_lead=None).runs[0].bold = True
    _bullets(doc, [
        (None, ("On-screen sentences that the narrator reads out or closely restates while they "
               "are on screen, for example a slide of full sentences read aloud.")),
        (None, ("Bullet points of five or more words that the narrator reads out, even when they "
               "are not full sentences, e.g. \u201cUnderstanding the purpose and scope of "
               "GDPR\u201d read aloud from a list. Shorter bullets (one to four words) are "
               "labels and do not count.")),
        (None, CAPTIONS[captions]),
    ])
    if captions == "pending":
        doc.paragraphs[-1].runs[-1].font.highlight_color = WD_COLOR_INDEX.YELLOW
        doc.paragraphs[-1].runs[-1].bold = True
    _para(doc, "Does NOT count as redundancy (these are signals that help, Mayer’s signaling "
               "principle), even when the narrator says the same words:").runs[0].bold = True
    _bullets(doc, [
        ("Labels, keywords, titles and headings, ", ("including numbering (“Step 1”) and "
                                                    "bullets of one to four words.")),
        ("Topic keywords repeated through the lecture, ",
         "e.g. “nuclear fusion”, “chain reaction”."),
        ("Insights: ", ("a short takeaway statement about a graph or diagram, e.g. “Alpha "
                       "increases with temperature” next to the graph, even when read word "
                       "for word.")),
        ("Aims and objectives, ", ("e.g. “In this video you will learn …”, "
                                  "“Objectives”, even when read aloud.")),
        ("Results and conclusions, ", ("e.g. “Result: …”, “Conclusion”, "
                                      "“Summary”, even when read aloud.")),
        ("One key statement highlighted in large type, ", ("and short prompts or buttons "
                                                          "(“Subscribe”, "
                                                          "“Try again”).")),
    ])
    _scale_table(doc, [
        ("1", "None: no on-screen text is read aloud, or only the signals listed above."),
        ("25", "Slight: a sentence or part of one is read aloud briefly."),
        ("50", ("Moderate: about half the time, on-screen sentences are read out or closely "
               "restated.")),
        ("75", "Substantial: most of the time, the on-screen sentences are read out."),
        ("100", ("Complete: the narration reads the on-screen sentences word for word "
                "throughout.")),
    ])

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    _heading(doc, "Coherence (Mayer): three parts")
    _para(doc, "People learn better when extraneous material is left out. Please judge three "
               "kinds of extraneous material. In the ‘Ratings’ sheet give one coherence "
               "score per segment covering all three; in ‘Video overall’ give the "
               "overall score and each part separately.")
    _para(doc, ("The intro (before the narrator starts speaking) and the outro (the closing "
                "sign-off, e.g. \u201csubscribe\u201d, \u201clink in the description\u201d, "
                "\u201csee you next time\u201d, and any end logo or credits) are not part of the "
                "lesson: they never count, for coherence or for redundancy. A channel or "
                "publisher logo on an intro or end card is fine; a large logo on a teaching "
                "slide is extraneous."), bold_lead="Intro and outro: ")
    _bullets(doc, [
        ("music_sounds, background music and sounds: ",
         ("music or sound effects playing under the narration. Music when nobody is speaking "
         "(e.g. an outro) does not count. You may take loudness into account; please say "
         "so in the notes if you do.")),
        ("offtopic_narration, off-topic talk: ",
         ("interesting but irrelevant stories, facts, jokes or remarks that are not needed to "
         "understand the lesson. Spoken signposts (“first”, “next”, "
         "“as a result”, “let’s move on”) are signals, not off-topic "
         "talk.")),
        ("onscreen_extraneous, extraneous on-screen material: ",
         ("decorative pictures or video clips unrelated to what is being taught (stock photos, "
         "mascots, unrelated clips), large logos, subscribe banners and promotional end "
         "screens shown while the narrator is teaching. Not extraneous: a small corner "
         "watermark; labels, headings and diagrams; pictures that show what is being "
         "explained.")),
    ])
    _scale_table(doc, [
        ("1", "None: everything on screen and in the audio serves the lesson."),
        ("25", "Minor: brief or faint, e.g. a few seconds of quiet music or one passing aside."),
        ("50", ("Noticeable: e.g. music under about half of the narration, a short digression, "
               "or a decorative picture on screen for a while.")),
        ("75", ("Substantial: e.g. music under most of the narration, repeated digressions, or "
               "decorative or promotional material on screen much of the time.")),
        ("100", "Pervasive: extraneous material throughout."),
    ])

    _heading(doc, "Instructor score (no action needed)")
    _para(doc, "The system also reports an instructor score = 100 − (redundancy + "
               "coherence) / 2, where higher means a better-designed video. You do not need to "
               "calculate it; it will be computed from your ratings in exactly the same way.")

    _heading(doc, "When you have finished")
    _para(doc, "Save your copy as ratings_<your name>.xlsx and send it back. If a second person "
               "rates, please work independently and do not compare ratings until both are "
               "done.")

    _heading(doc, "Questions for you (from the student)")
    _bullets(doc, [
        ("Second rater: ", ("could a second person rate at least some of the videos "
                           "independently? Their agreement with you shows how much any system "
                           "could be expected to agree with people.")),
        ("Intros and outros: ", ("should a title card or end screen with no narration lower a "
                                "video’s redundancy, or count as nothing redundant?")),
        ("Text-free videos: ", ("a video with narration but no on-screen text: the lowest "
                               "redundancy (1), or not applicable (leave empty)?")),
        ("Timeline: ", "when can the ratings be expected, and when is the thesis due?"),
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--captions", choices=sorted(CAPTIONS), required=True)
    a = ap.parse_args()
    print(build(a.out, a.captions))


if __name__ == "__main__":
    main()
