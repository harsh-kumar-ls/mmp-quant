# Data directory

```
data/
  videos/            # input videos for `mmpquant validate`
  manual_scores.csv  # professor's ground truth — NOT YET AVAILABLE
```

## manual_scores.csv

**This file does not exist yet and must not be created by any tool, script, or
assistant.** No fabricated, simulated, or placeholder scores. Ever.

Expected format (to be **confirmed with the professor** when the real file
arrives — the assumption below is unverified):

```csv
video_id,manual_score
lecture_01,72
lecture_02,15
```

`video_id` = video filename without extension.

Unresolved before any tuning can begin:

- What scale is `manual_score` on? 0–100? 1–5 Likert? Rank order?
- What criteria did the rater apply — is it "redundancy" in Mayer's exact
  sense, or general perceived quality?
- How many videos will be scored? (`n` drives whether Spearman ≥ 0.7 is a
  meaningful result or noise — at n=10 the 95% CI around ρ=0.7 is roughly
  [0.11, 0.92].)
- One rater or more? A single rater gives no inter-rater reliability estimate,
  so we cannot separate "our system is wrong" from "the ground truth is noisy".

Note: Spearman is rank-based, so a scale mismatch between our 0–100 score and
the professor's scale is harmless. A **criteria** mismatch is not.
