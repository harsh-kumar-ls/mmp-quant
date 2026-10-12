#!/bin/bash
# Re-score the videos named as arguments (cached ASR/OCR reused). Appends to
# runs/score_batch_oct05_summary.txt. Split into small lists so each call ends
# inside the harness's background time limit.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
# Every model is cached locally; skip the Hugging Face network checks that
# retried for minutes on a flaky connection (08 Oct 2026).
export HF_HUB_OFFLINE=1
SUMMARY="runs/score_batch_oct05_summary.txt"
for name in "$@"; do
  start=$(date +%s)
  if .venv/bin/mmpquant score "data/videos/${name}.mp4" -o "runs/${name}" > "runs/${name}.log" 2>&1; then
    s=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${s}  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  fi
done
