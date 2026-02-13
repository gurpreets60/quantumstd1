#!/usr/bin/env bash
# Full-dataset quantum LSTM + winning sklearn ensemble, running indefinitely.
# Quantum uses all 20,315 train / 2,555 val / 3,720 test samples.
# Each cycle: train all models → save models → CFA → repeat.
# Stop with: ./prog_full.sh stop
set -e

PYTHON=".venv/bin/python"
SCRIPT="pred_lstm.py"
START=$(date +%s)
RUN=0
PIDFILE="data/.full.pid"

# Quantum LSTM parameters (full dataset)
QLSTM_INPUT=3       # compress to 3 dims
QLSTM_HIDDEN=2      # 5 qubits total
QLSTM_DEPTH=2       # VQC depth 2
QLSTM_EPOCHS=5      # 5 epochs per cycle
# Budget set very high so auto-calibration uses all samples.
# Actual runtime is ~2-3 hours per cycle (not 999999s).
QLSTM_BUDGET=999999

# Winning CFA ensemble + QUANTUM LSTM
WINNERS="PERCEPTRON,PASSIVE AGGRESSIVE,BAGGING DT2,QUANTUM LSTM"

echo $$ > "$PIDFILE"
trap 'echo ""; echo "=== Stopped after cycle $RUN ($(( $(date +%s) - START ))s elapsed) ==="; rm -f "$PIDFILE"; exit 0' INT TERM EXIT

N_QUBITS=$((QLSTM_INPUT + QLSTM_HIDDEN))
echo "=== Full-Dataset Quantum Training (indefinite) ==="
echo "Start: $(date)"
echo "Quantum: ${N_QUBITS} qubits, depth=${QLSTM_DEPTH}, ${QLSTM_EPOCHS} epochs/cycle"
echo "Dataset: full 20,315 train / 2,555 val / 3,720 test"
echo "CFA filter: ${WINNERS}"
echo "Stop with: ./prog_full.sh stop"
echo ""

while true; do
    RUN=$((RUN + 1))
    ELAPSED=$(( $(date +%s) - START ))

    echo ""
    echo "############################################"
    echo "  CYCLE $RUN — $(date) (elapsed: ${ELAPSED}s)"
    echo "############################################"

    $PYTHON $SCRIPT -o train -m all \
        --time_limit 999999 \
        --mem_limit 0 \
        -qe "$QLSTM_EPOCHS" \
        -qi "$QLSTM_INPUT" \
        -qh "$QLSTM_HIDDEN" \
        -qd "$QLSTM_DEPTH" \
        -qt "$QLSTM_BUDGET" \
        -e 0 \
        --cfa_time 30 \
        --cfa_models "$WINNERS" \
        2>&1 | tee -a "data/full_log_$(date +%Y%m%d).txt"

    echo ""
    LATEST_RUN=$(ls -td data/run_* | head -1)
    echo "[CYCLE $RUN DONE] $(date) — Results: $LATEST_RUN"

    ELAPSED=$(( $(date +%s) - START ))
    echo "Total elapsed: ${ELAPSED}s"
done
