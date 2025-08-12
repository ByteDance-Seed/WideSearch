#!/bin/bash

set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installation completed"
    # Add uv to PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python
uv python install 3.11
uv python pin 3.11

# Initialize virtual environment
echo "> Creating virtual environment..."
uv venv --prompt widesearch || {
    echo "✗ Failed to create virtual environment"
    exit 1
}


# Activate environment
source .venv/bin/activate
uv sync

echo "✓ Environment setup completed"
echo "Run 'source .venv/bin/activate' to activate the environment"