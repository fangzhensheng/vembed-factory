#!/bin/bash

# Visual RAG Environment Setup Script

echo "Select setup method:"
echo "1. uv (Recommended for local)"
echo "2. Conda"
echo "3. Docker"
read -p "Enter choice [1-3]: " choice

if [ "$choice" = "1" ]; then
    echo "Installing via uv..."
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true
    fi

    # Sync project (creates .venv, installs deps + dev tools, respects uv.lock)
    echo "Syncing project with uv..."
    uv sync

    echo ""
    echo "Setup complete!"
    echo "  Activate manually : source .venv/bin/activate"
    echo "  Or run via uv     : uv run python train.py"
    echo "  Install all extras: uv sync --all-extras"

elif [ "$choice" = "2" ]; then
    echo "Creating Conda environment 'visual-rag'..."
    if ! command -v conda &> /dev/null; then
        echo "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    
    conda env create -f environment.yaml
    echo "Setup complete. Activate with: conda activate visual-rag"

elif [ "$choice" = "3" ]; then
    echo "Building Docker image..."
    docker build -t vembed-factory .
    echo ""
    echo "Setup complete!"
    echo "  Run interactive shell : docker run --gpus all -it -v \$(pwd)/data:/app/data vembed-factory bash"
    echo "  Or use docker compose : docker compose up -d"

else
    echo "Invalid choice."
    exit 1
fi
