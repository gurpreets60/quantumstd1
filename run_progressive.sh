#!/usr/bin/env bash
# Progressive time-limit training: time allowance grows as runtime increases.
# Runs indefinitely until stopped (Ctrl+C / kill).
# Results accumulate in data/run_* directories for comparison.
set -e

PYTHON=".venv/bin/python"
SCRIPT="pred_lstm.py"
START=$(date +%s)
RUN=0

# Expanding time limits: walk through this list, then repeat the last value.
# Covers the useful range; beyond 3600s sklearn models use full data anyway.
TIME_LIMITS=(1 2 5 10 20 30 60 120 300 600 1200 1800 3600)
MAX_IDX=$(( ${#TIME_LIMITS[@]} - 1 ))
IDX=0

trap 'echo ""; echo "=== Stopped after run $RUN, time_limit=${TL}s ($(( $(date +%s) - START ))s elapsed) ==="; exit 0' INT TERM

echo "=== Progressive Training (expanding time limits) ==="
echo "Start: $(date)"
echo "Time limits: ${TIME_LIMITS[*]} (then repeats max)"
echo "Stop with Ctrl+C"
echo ""

while true; do
    RUN=$((RUN + 1))
    TL=${TIME_LIMITS[$IDX]}
    ELAPSED=$(( $(date +%s) - START ))

    echo ""
    echo "=========================================="
    echo "  run=$RUN  time_limit=${TL}s  (elapsed: ${ELAPSED}s)"
    echo "=========================================="

    # CFA time scales with time limit
    CFA_TIME=$(( TL > 10 ? 30 : 10 ))

    $PYTHON $SCRIPT -o train -m sklearn \
        --time_limit "$TL" \
        --cfa_time "$CFA_TIME" \
        2>&1 | tee -a "data/progressive_log_$(date +%Y%m%d).txt"

    echo ""
    echo "[RUN DONE] run=$RUN time_limit=${TL}s completed at $(date)"

    LATEST_RUN=$(ls -td data/run_* | head -1)
    if [ -f "$LATEST_RUN/cfa_greedy.csv" ]; then
        echo "[SUMMARY] Results in: $LATEST_RUN"
    fi

    # Advance to next time limit, stay at max once reached
    if [ "$IDX" -lt "$MAX_IDX" ]; then
        IDX=$((IDX + 1))
    fi
done
