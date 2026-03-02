#!/usr/bin/env bash
# Train only the two batch quantum models (QLSTM + VQFWP).
# Usage: ./run_quantum_batch.sh [epochs] [qbatch_time_budget] [model_time_limit]
set -e

PYTHON=".venv/bin/python"
SCRIPT="pred_lstm.py"
EPOCHS="${1:-1}"
QB_TIME="${2:-8}"
MODEL_TL="${3:-90}"

echo "=== Batch Quantum Models ==="
echo "Epochs: ${EPOCHS}"
echo "Per-model quantum time budget: ${QB_TIME}s"
echo "Per-model guard time limit: ${MODEL_TL}s"
echo ""

$PYTHON $SCRIPT -o train -m quantum_batch \
    --qbatch_epoch "$EPOCHS" \
    --qbatch_time "$QB_TIME" \
    --time_limit "$MODEL_TL" \
    --cfa_time 5 \
    2>&1 | tee "data/quantum_batch_log_$(date +%Y%m%d_%H%M%S).txt"

LATEST_RUN=$(ls -td data/run_* | head -1)
echo ""
echo "=== Done ==="
echo "Results: $LATEST_RUN"
