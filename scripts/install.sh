#!/bin/bash

# Default to system-wide installation
use_system="--system"

# Check for command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --venv | --use-venv)
        use_system="" # Empty string when using venv
        ;;
    --system | --no-venv)
        use_system="--system" # Explicitly set to system
        ;;
    *)
        echo "Unknown parameter: $1"
        echo "Usage: $0 [--venv|--system]"
        exit 1
        ;;
    esac
    shift
done

# Main packages installation
uv pip install $use_system --no-cache-dir -r requirements.txt
uv pip install $use_system --no-cache-dir -r "requirements-${SYSTEM_VARIANT:-cuda}.txt"

# Install development packages
uv pip install $use_system hatchling editables
uv pip install $use_system -e .

# Uninstall flash-attn if we're on GH200
if [[ "$SYSTEM_VARIANT" == "gh200" ]]; then
    echo "Running on GH200, uninstalling flash-attn..."
    uv pip uninstall $use_system flash-attn
else
    echo "Not running on GH200, uninstalling asyncio..."
    uv pip uninstall $use_system asyncio
fi