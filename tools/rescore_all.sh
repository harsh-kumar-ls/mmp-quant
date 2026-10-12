#!/bin/bash
# Re-run every scored video with --no-resume after the defect 9.1 fix.
#
# The fix changes only a reported statistic (detection share is now weighted by
# duration rather than by dedup group), not the score itself. The re-run exists
# so the stored artifacts -- which are the thesis audit trail -- carry the
# corrected number rather than the misleading one.
#
# ASR is re-run too, since there is no per-stage force flag. Scores are compared
# against a pre-run snapshot afterwards; any drift is ASR nondeterminism and
# must be reported, not quietly absorbed.

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"

SUMMARY="runs/rescore_summary.txt"
: > "$SUMMARY"

VIDEOS=(
  "14_JzbhQX00" "test4" "test6" "test8" "test9" "test_1" "test_10" "test_2" "test_3" "test_7"
  "Test_11" "Test_12" "Test_13" "Test_14" "Test_15" "Test_16" "Test_17" "Test_17_2"
  "Test_18" "Test_19" "Test_20" "Test_21" "Test_22" "Test_23"
)

echo "rescore started $(date)" >> "$SUMMARY"
for name in "${VIDEOS[@]}"; do
  start=$(date +%s)
  if .venv/bin/mmpquant score "data/videos/${name}.mp4" -o "runs/${name}" --no-resume \
       > "runs/${name}.log" 2>&1; then
    s=$(grep -oE "redundancy: (N/A|[0-9.]+)" "runs/${name}.log" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE   $name  ${s}  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  else
    echo "[$(date '+%H:%M:%S')] FAILED $name  ($(( $(date +%s) - start ))s)" | tee -a "$SUMMARY"
  fi
done
echo "rescore finished $(date)" >> "$SUMMARY"
