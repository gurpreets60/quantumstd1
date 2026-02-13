#!/usr/bin/env bash
# Train Quantum LSTM + winning sklearn ensemble, then CFA.
# Quantum circuit size scales with time budget to maximize samples.
#
# Usage: ./run_quantum.sh [quantum_time_budget_seconds]
#   ./run_quantum.sh 1      # debug (3 qubits, 1 epoch)
#   ./run_quantum.sh 10     # short (4 qubits, 3 epochs)
#   ./run_quantum.sh 120    # medium (5 qubits, 10 epochs)
#   ./run_quantum.sh 3600   # full (6 qubits, 20 epochs, needs ~1hr)
set -e

PYTHON=".venv/bin/python"
SCRIPT="pred_lstm.py"
QT="${1:-1}"  # quantum time budget in seconds

# Winning CFA ensemble + QUANTUM LSTM
WINNERS="PERCEPTRON,PASSIVE AGGRESSIVE,BAGGING DT2,QUANTUM LSTM"

# Scale quantum circuit complexity with time budget.
# More qubits = more expressive but ~exponentially slower per sample.
if [ "$QT" -ge 600 ]; then
    QLSTM_INPUT=3; QLSTM_HIDDEN=3; QLSTM_DEPTH=2; QLSTM_EPOCHS=20  # 6 qubits
elif [ "$QT" -ge 60 ]; then
    QLSTM_INPUT=3; QLSTM_HIDDEN=2; QLSTM_DEPTH=2; QLSTM_EPOCHS=10  # 5 qubits
elif [ "$QT" -ge 10 ]; then
    QLSTM_INPUT=2; QLSTM_HIDDEN=2; QLSTM_DEPTH=1; QLSTM_EPOCHS=3   # 4 qubits
else
    QLSTM_INPUT=2; QLSTM_HIDDEN=1; QLSTM_DEPTH=1; QLSTM_EPOCHS=1   # 3 qubits
fi

N_QUBITS=$((QLSTM_INPUT + QLSTM_HIDDEN))

# Sklearn time limit: generous so winners train on full 20k samples
SKLEARN_TL=30

# Overall time limit for quantum (TimeGuard kill ceiling)
MODEL_TL=$(( QT + 120 ))

echo "=== Quantum LSTM + Winning Ensemble ==="
echo "Quantum budget: ${QT}s"
echo "Quantum: ${N_QUBITS} qubits (in=${QLSTM_INPUT} hid=${QLSTM_HIDDEN}), depth=${QLSTM_DEPTH}, epochs=${QLSTM_EPOCHS}"
echo "Sklearn: ${SKLEARN_TL}s time limit (full 20k samples)"
echo "CFA filter: ${WINNERS}"
echo ""

$PYTHON $SCRIPT -o train -m all \
    --time_limit "$MODEL_TL" \
    -qe "$QLSTM_EPOCHS" \
    -qi "$QLSTM_INPUT" \
    -qh "$QLSTM_HIDDEN" \
    -qd "$QLSTM_DEPTH" \
    -qt "$QT" \
    -e 0 \
    --cfa_time 30 \
    --cfa_models "$WINNERS" \
    2>&1 | tee "data/quantum_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
LATEST_RUN=$(ls -td data/run_* | head -1)
echo "=== Done ==="
echo "Results: $LATEST_RUN"
echo ""
echo "Re-run CFA (sklearn-only, no truncation):"
echo "  $PYTHON $SCRIPT -o cfa --run_dir $LATEST_RUN --cfa_models 'PERCEPTRON,PASSIVE AGGRESSIVE,BAGGING DT2'"
echo "Re-run CFA (with quantum):"
echo "  $PYTHON $SCRIPT -o cfa --run_dir $LATEST_RUN --cfa_models '$WINNERS'"
