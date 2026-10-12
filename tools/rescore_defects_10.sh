#!/bin/bash
# Re-score every video after fixing defects 10.1 (classifier fallback) and
# 10.2 (aggregation formula). Both fixes take effect downstream of the cached
# ASR/OCR artifacts (classification happens in align.py's window-building,
# scoring happens after that), so --no-resume is NOT needed here: reusing
# transcript.json/ocr.json and only re-running alignment + scoring + report
# generation is correct and far faster.

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

SUMMARY="runs/rescore_sentence_gate_summary.txt"
: > "$SUMMARY"

VIDEOS=(
  "14_JzbhQX00" "test4" "test6" "test8" "test9" "test_1" "test_10" "test_2" "test_3" "test_7"
  "Test_11" "Test_12" "Test_13" "Test_14" "Test_15" "Test_16" "Test_17" "Test_17_2"
  "Test_18" "Test_19" "Test_20" "Test_21" "Test_22" "Test_23"
)

echo "rescore started $(date)" >> "$SUMMARY"
for name in "${VIDEOS[@]}"; do
  start=$(date +%s)
  if .venv/bin/mmpquant score "data/videos/${name}.mp4" -o "runs/${name}" \
       > "runs/${name}.log" 2>&1; then
    s=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${s}  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  fi
done
echo "rescore finished $(date)" >> "$SUMMARY"
