# MMP-Quant

Scoring educational videos against two of **Mayer's multimedia principles**:

| Principle | What is measured | Score |
|---|---|---|
| **Redundancy** | on-screen text that the narration restates at the same moment | 0–100, higher = worse |
| **Coherence** | background music playing under the narration (off-topic narration is measured too, as an experimental part kept out of the score) | 0–100, higher = worse |

A B.Tech research project. Every score comes with its evidence (the actual on-screen
text and narration, the seconds of music, and the arithmetic), so a reader can check
any number by hand.

**Status:** the extraction pipeline, both scorers, the Markdown report and a local web UI are
built and tested. The validation harness (`mmpquant validate`, correlation against human
ratings) is deliberately not built yet: it waits for the real manual scores, because a
stub fed with placeholder scores would produce numbers that look real and are not.

## Setup

Requires Python 3.11+ and ffmpeg. Developed on macOS (the default OCR backend uses
Apple's Vision framework; set `ocr.backend: rapidocr` in `config.yaml` elsewhere).

```bash
brew install ffmpeg python@3.11 uv
uv venv --python "$(brew --prefix python@3.11)/bin/python3.11" .venv
uv pip install --python .venv/bin/python -e '.[dev,webapp]'
uv pip install --python .venv/bin/python "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
```

The first run downloads model weights: Whisper `small`, MiniLM sentence embeddings, and
the AudioSet sound classifier (about 1 GB in total).

## Use

```bash
mmpquant config-check                 # validate config.yaml
mmpquant score  path/to/video.mp4     # extract, score both principles, write the report
mmpquant report runs/<video_id>       # re-render report.md from a run
mmpquant serve                        # web UI at http://127.0.0.1:8000
```

Each run writes to `runs/<video_id>/`:

| file | contents |
|---|---|
| `transcript.json` | narration with word-level timestamps (faster-whisper) |
| `ocr.json` | on-screen text, de-duplicated across static frames |
| `audio_events.json` | sound classes heard, second by second (coherence) |
| `windows.json` | narration and on-screen text aligned per 10-second window |
| `result.json` | every window's score with its evidence, both principles, and the full config used |
| `report.md` | human-readable report: scores, segment timelines, worst segments, methodology |
| `segment_scores.csv`, `coherence_segment_scores.csv` | per-segment scores for comparison with human ratings |

Re-running reuses the saved transcript, OCR and audio events, so changing a scoring
parameter re-scores in about a minute. Use `--no-resume` to recompute them.

## How the scores work

Both principles use 10-second windows, and both roll windows up the same way: a
duration-weighted mean over **all** windows, where a window with nothing to assess counts
as 0. The video score is therefore exactly the mean of its 30-second segment scores, and
does not depend on where the segment boundaries fall.

**Redundancy:** only on-screen text classified as a full sentence counts; titles, labels
and key terms are signaling, not redundancy. A window scores above 0 only if the narration
actually restates an on-screen sentence, meaning it says more than half of the sentence's
content words, and more of them than narration elsewhere in the video does by chance.
Above that gate, two measures are taken, each above chance, and combined 0.6 / 0.4: the
share of the on-screen words the narration says (in order), and how close in meaning each
on-screen sentence is to the narration sentence that restates it. Captions that run one
sentence across several screens are joined back into whole sentences first.

**Coherence:** the share of spoken seconds with music under them, according to an AudioSet
sound classifier (`MIT/ast-finetuned-audioset-10-10-0.4593`). Music with nobody speaking,
such as an intro jingle, is recorded but not counted. Off-topic narration (a window's
narration being nearer in meaning to unrelated everyday text than to the video's own
topic) is computed and shown but carries weight 0. Read by hand, most windows it flagged
were short remarks about the board rather than real digressions.

A video with no narration at all scores **N/A, not 0**: it could not be assessed.

Every threshold and weight is in [`config.yaml`](config.yaml), with the reasoning
behind each value written next to it. None has been tuned against human ratings yet.

## Tests

```bash
.venv/bin/python -m pytest     # 432 tests, about 10 s, no model downloads
```

The scorers take their models by dependency injection, so the scoring rules are tested
against hand-computed values with synthetic text, transcripts and sound timelines.

## Architecture

```
video
 ├─ ffmpeg → audio ─┬─ faster-whisper ─────────────→ transcript.json
 │                  └─ AudioSet classifier ─────────→ audio_events.json
 └─ ffmpeg → frames → frame-diff dedup → OCR ───────→ ocr.json
                                   │
              alignment (10 s windows, ±3 s tolerance) → windows.json
                                   │
           PrincipleScorer plugins ├─ redundancy.py
                                   └─ coherence.py  → result.json → report.md / web UI
```

- **Each extraction stage runs in its own process.** This avoids native-library clashes
  between the speech and OCR stacks, frees each model's memory when its stage ends, and
  makes every stage independently resumable and inspectable.
- **Scoring is a plugin.** Each principle is one `PrincipleScorer` subclass that reads
  the shared windows (plus a per-video context of transcript and sound timeline).

## Notes

- Videos and run outputs are not in this repository (`data/videos/` and `runs/` are
  ignored): the sample videos belong to their creators.
- Code comments sometimes cite `PROGRESS.md`, the author's private research log, for the
  history behind a design decision. It is not included here.
