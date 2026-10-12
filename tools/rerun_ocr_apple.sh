#!/bin/bash
# Re-run OCR for every video with the Apple Vision backend (04 Oct 2026).
#
# The RapidOCR output is preserved as runs/<id>/rapidocr_baseline/ocr.json,
# not overwritten, so the two engines can be compared on identical frames.
# ASR is reused (transcript.json kept); only ocr.json is regenerated, then the
# video is re-scored. Sequential, for the reason in batch_score.sh.

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
SUMMARY="runs/rerun_ocr_apple_summary.txt"
: > "$SUMMARY"
VIDEOS=(
  "14_JzbhQX00" "test4" "test6" "test8" "test9" "test_1" "test_10" "test_2" "test_3" "test_7"
  "Test_11" "Test_12" "Test_13" "Test_14" "Test_15" "Test_16" "Test_17" "Test_17_2"
  "Test_18" "Test_19" "Test_20" "Test_21" "Test_22" "Test_23"
)
echo "started $(date)" >> "$SUMMARY"
for name in "${VIDEOS[@]}"; do
  dir="runs/${name}"
  start=$(date +%s)
  if [ ! -f "$dir/rapidocr_baseline/ocr.json" ]; then
    mkdir -p "$dir/rapidocr_baseline"
    cp "$dir/ocr.json" "$dir/rapidocr_baseline/ocr.json"
  fi
  # Skip OCR if this video already has Apple output (the run is resumable).
  if [ -f "$dir/ocr.json" ] && grep -q '"backend": "apple"' "$dir/ocr.json"; then
    :
  else
    rm -f "$dir/ocr.json"
  fi
  if .venv/bin/mmpquant score "data/videos/${name}.mp4" -o "$dir" > "runs/${name}.log" 2>&1; then
    s=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${s}  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  fi
done
echo "finished $(date)" >> "$SUMMARY"
