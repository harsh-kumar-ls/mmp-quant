#!/bin/bash
# Score the videos added on 09 Sep 2026 (Test_11 .. Test_23).
#
# Same sequential design as batch_score.sh: one video's ASR step alone
# saturates ~4-5 cores, so parallelism would only cause contention.
# Continues past a failing video so one bad file doesn't block the batch.

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

SUMMARY="runs/batch_summary_new.txt"
mkdir -p runs

VIDEOS=(
  "Test_11" "Test_12" "Test_13" "Test_14" "Test_15" "Test_16"
  "Test_17" "Test_17_2" "Test_18" "Test_19" "Test_20"
  "Test_21" "Test_22" "Test_23"
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
