#!/bin/bash
# Score a list of videos one at a time, writing a running plain-text summary.
#
# Deliberately sequential, not parallel: each video's ASR step alone saturates
# ~4-5 CPU cores on this machine, so running several at once would only make
# them contend for the same cores rather than finish faster.
#
# Continues past a failing video rather than aborting the whole batch, so one
# bad file doesn't block the rest of an hours-long run.

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

SUMMARY="runs/batch_summary.txt"
mkdir -p runs

# Corpus for supervisor-directed validation (2-3 minute clips, mixed styles).
# Each video also gets a blank annotation sheet written to data/annotations/.
VIDEOS=(
  "14_JzbhQX00"
  "test4"
  "test6"
  "test8"
  "test9"
  "test_1"
  "test_10"
  "test_2"
  "test_3"
  "test_7"
)

echo "batch started $(date)" >> "$SUMMARY"

for name in "${VIDEOS[@]}"; do
  video="data/videos/${name}.mp4"
  start=$(date +%s)
  echo "[$(date '+%H:%M:%S')] START  $name" | tee -a "$SUMMARY"

  if .venv/bin/mmpquant score "$video" -o "runs/${name}" > "runs/${name}.log" 2>&1; then
    .venv/bin/mmpquant annotation-sheet "$video" >> "runs/${name}.log" 2>&1
    score_line=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    elapsed=$(( $(date +%s) - start ))
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${score_line}  (${elapsed}s)" | tee -a "$SUMMARY"
  else
    elapsed=$(( $(date +%s) - start ))
    echo "[$(date '+%H:%M:%S')] FAILED $name  see runs/${name}.log  (${elapsed}s)" | tee -a "$SUMMARY"
  fi
done

echo "batch finished $(date)" >> "$SUMMARY"
