---
title: MMP-Quant
emoji: 📼
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Scores educational videos on Mayer's redundancy and coherence
---

# MMP-Quant

A B.Tech research project. Upload a short educational video (about 3 minutes
works best) and the system scores it on two of Mayer's multimedia principles:

- **Redundancy** (0–100, higher = worse): how much of the time the on-screen
  text is read aloud word for word, including burned-in captions. Labels,
  keywords, titles, aims, results and short insights are signals, not redundancy.
- **Coherence** (0–100, higher = worse): background music under the narration.
  The first 10 seconds (the intro) never count.
- **Instructor score** = 100 − (redundancy + coherence) / 2, higher = better.

Every report shows the evidence: the on-screen text and the narration side by
side, per 10-second window and per 30-second segment.

**Two kinds of reports.** The library's reports were scored on a Mac with Apple
Vision OCR; those are the numbers in the thesis. Videos uploaded here are scored
in the cloud with RapidOCR, which reads handwriting and chalk less well, so their
scores can differ a little. Each report says which.

Scoring takes several minutes on this free server, which also sleeps after two
days without visitors (the first visit then takes about a minute). Uploaded
videos are not kept after the server restarts.
