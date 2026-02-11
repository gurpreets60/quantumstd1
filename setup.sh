#!/usr/bin/env bash
set -e

PYTHON_VERSION=3.13

# Install Python if needed
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/"
    exit 1
fi

echo "=== Setting up quantumstd1 ==="

# Create venv if missing
if [ ! -d .venv ]; then
    echo "Installing Python $PYTHON_VERSION..."
    uv python install "$PYTHON_VERSION"
    echo "Creating venv..."
    uv venv --python "$PYTHON_VERSION" .venv
fi

# Install all dependencies
echo "Installing dependencies..."
uv pip install --python .venv/bin/python \
    tensorflow tf-keras scikit-learn scipy numpy \
    psutil rich pennylane torch pandas

# Clone CFA module if not present
CFA_DIR="$HOME/projects/quantumfusion/inital_code/unified_cfa"
if [ ! -f "$CFA_DIR/cfa.py" ]; then
    echo ""
    echo "WARNING: CFA module not found at $CFA_DIR/cfa.py"
    echo "CFA fusion will fail. Place cfa.py at the path above."
fi

echo ""
echo "=== Setup complete ==="
echo "Run: uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 1 -e 1 --time_limit 30"
