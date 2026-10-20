#!/bin/bash
# The regression loop (20 Oct 2026). Run before reporting ANY change as done.
#
#   tools/loop_check.sh          full: re-scores every video (~10-20 min)
#   tools/loop_check.sh --quick  checks the existing runs/ without re-scoring
#
# Steps, in order of cost; every step runs even if an earlier one fails, so
# one run shows everything that is wrong:
#   1. lint                      ruff
#   2. unit tests                pytest            (code does what was intended)
#   3. browser tests             pytest -m e2e     (the web UI, as a user uses it)
#   4. regression                mmpquant regress  (the user's expectations hold;
#                                                   every score change listed)
# Exit 0 only if all pass and no score changed. Exit 2 = all pass but scores
# changed: review runs/_regress/report.md, then `mmpquant regress --accept`.
set -u
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
results=()

step() {  # name, command...
  local name="$1"; shift
  echo; echo "=== $name"
  "$@"
  local code=$?
  results+=("$name:$code")
  return 0
}

step "lint" .venv/bin/ruff check src tests
step "unit tests" $PY -m pytest -p no:warnings
step "browser tests" $PY -m pytest -p no:warnings -m e2e tests/e2e
if [ "${1:-}" = "--quick" ]; then
  step "regression (existing runs)" .venv/bin/mmpquant regress --from-runs
else
  step "regression (full re-score)" .venv/bin/mmpquant regress
fi

echo; echo "=== summary"
worst=0
for r in "${results[@]}"; do
  name="${r%%:*}"; code="${r##*:}"
  case "$code" in
    0) echo "  PASS  $name" ;;
    2) echo "  REVIEW $name — scores changed; see runs/_regress/report.md"; [ $worst -lt 2 ] && worst=2 ;;
    *) echo "  FAIL  $name (exit $code)"; worst=1 ;;
  esac
done
# a FAIL outranks a REVIEW
for r in "${results[@]}"; do c="${r##*:}"; [ "$c" != 0 ] && [ "$c" != 2 ] && worst=1; done
exit $worst
