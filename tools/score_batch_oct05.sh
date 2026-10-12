#!/bin/bash
# 05 Oct 2026: re-score the 24-video corpus under the question ruling (cached
# ASR/OCR reused), then fully score the four new videos and write their blank
# two-rater annotation sheets. Sequential, as in batch_score.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
SUMMARY="runs/score_batch_oct05_summary.txt"
: > "$SUMMARY"
EXISTING=( "14_JzbhQX00" "test4" "test6" "test8" "test9" "test_1" "test_10" "test_2" "test_3" "test_7"
  "Test_11" "Test_12" "Test_13" "Test_14" "Test_15" "Test_16" "Test_17" "Test_17_2"
  "Test_18" "Test_19" "Test_20" "Test_21" "Test_22" "Test_23" )
NEW=( "Test_24" "Test_25" "test_26" "Test_27" )
run() {
  local name="$1"; local start=$(date +%s)
  if .venv/bin/mmpquant score "data/videos/${name}.mp4" -o "runs/${name}" > "runs/${name}.log" 2>&1; then
    s=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${s}  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  fi
}
echo "started $(date)" >> "$SUMMARY"
for n in "${NEW[@]}"; do run "$n"; .venv/bin/mmpquant annotation-sheet "data/videos/${n}.mp4" >> "runs/${n}.log" 2>&1; done
for n in "${EXISTING[@]}"; do run "$n"; done
echo "finished $(date)" >> "$SUMMARY"
