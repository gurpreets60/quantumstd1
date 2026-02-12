#!/usr/bin/env bash
# Progressive time-limit training: increasing time allowances over 10 hours.
# Each run trains all sklearn models with a larger time budget, then runs CFA.
# Results accumulate in data/run_* directories for comparison.
set -e

PYTHON=".venv/bin/python"
SCRIPT="pred_lstm.py"
START=$(date +%s)
TOTAL_SECS=$((10 * 3600))  # 10 hours

# Time limits to try (seconds), progressively increasing.
# Models auto-calibrate training samples to fit each budget.
TIME_LIMITS=(1 2 5 10 20 30 60 120 300 600 1200 1800 3600)

echo "=== Progressive Time-Limit Training ==="
echo "Start: $(date)"
echo "Budget: 10 hours (${TOTAL_SECS}s)"
echo "Time limits: ${TIME_LIMITS[*]}"
echo ""

for TL in "${TIME_LIMITS[@]}"; do
    ELAPSED=$(( $(date +%s) - START ))
    REMAINING=$(( TOTAL_SECS - ELAPSED ))

    if [ "$REMAINING" -le 0 ]; then
        echo "[DONE] 10-hour budget exhausted after ${ELAPSED}s"
        break
    fi

    # Estimate if this run can finish. Most sklearn models on this 20k-sample
    # dataset finish in <30s regardless of time_limit (it's a kill ceiling,
    # not actual runtime). At most a few slow models might approach time_limit.
    EST_RUN=$(( 3 * TL + 300 ))
    if [ "$EST_RUN" -gt "$REMAINING" ]; then
        echo "[SKIP] time_limit=${TL}s would take ~${EST_RUN}s, only ${REMAINING}s left"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  time_limit=${TL}s  (elapsed: ${ELAPSED}s / ${TOTAL_SECS}s)"
    echo "=========================================="

    # CFA time scales with model count and time limit
    CFA_TIME=$(( TL > 10 ? 30 : 10 ))

    $PYTHON $SCRIPT -o train -m sklearn \
        --time_limit "$TL" \
        --cfa_time "$CFA_TIME" \
        2>&1 | tee -a "data/progressive_log_$(date +%Y%m%d).txt"

    echo ""
    echo "[RUN DONE] time_limit=${TL}s completed at $(date)"

    # Brief summary of this run's best CFA result
    LATEST_RUN=$(ls -td data/run_* | head -1)
    if [ -f "$LATEST_RUN/cfa_greedy.csv" ]; then
        echo "[SUMMARY] Results in: $LATEST_RUN"
    fi
done

echo ""
echo "=== All Progressive Runs Complete ==="
echo "Total elapsed: $(( $(date +%s) - START ))s"
echo "Run directories:"
ls -td data/run_* | head -20
